from activity_generation_endpoint.objects.activity_generation.activity_generation_model import ActivityGenerationModel
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from ACI_AI_Backend.objects.mutex_lock import lock
from huggingface_hub import hf_hub_download
from unsloth import is_bfloat16_supported
from django.conf import settings
from datasets import Dataset
from trl import SFTTrainer
import torch
import json
import os
import re
import gc

class ActivityGenerator:
    def __init__(self):
        file = open(settings.ACTIVITY_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)

        self.instruction = config["instruction"]
        self.prompt_backbone = """
        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
        file.close()
    
    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()
    
    def generate_activity(self, case_title, case_description, task_title, description):
        with lock:
            self.cleanup()
            if ActivityGenerationModel.model is None or ActivityGenerationModel.tokenizer is None:
                ActivityGenerationModel.load()

            input_string = f"Case title: {case_title}\nCase description: {case_description}\nTask title: {task_title}\nTask description: {description}"
            inputs = ActivityGenerationModel.tokenizer(
            [
                self.prompt_backbone.format(
                    self.instruction,
                    input_string,
                    "",
                )
            ], return_tensors = "pt").to("cuda")
            
            tags = ["<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>"]
            
            outputs = ActivityGenerationModel.model.generate(**inputs, max_new_tokens = 300, use_cache = True)
            output_text = ActivityGenerationModel.tokenizer.batch_decode(outputs)[0]
            
            for tag in tags:
                output_text = output_text.replace(tag, "")
            
            response_search = re.search("### Response:\n", output_text)
            response = output_text[response_search.start(): ].replace("### Response:\n", "")
            
            self.cleanup()
            return response
            
activity_generator = ActivityGenerator()