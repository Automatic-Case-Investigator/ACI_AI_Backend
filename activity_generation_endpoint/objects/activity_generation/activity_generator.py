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
    
    def generate_activity(self, case_title, task_title, description):
        with lock:
            if ActivityGenerationModel.model is None or ActivityGenerationModel.tokenizer is None:
                ActivityGenerationModel.load()

            input_string = f"Case title: {case_title}\nTask title: {task_title}\nDescription: {description}"
            inputs = ActivityGenerationModel.tokenizer(
            [
                self.prompt_backbone.format(
                    self.instruction,
                    input_string,
                    "",
                )
            ], return_tensors = "pt").to("cuda")
            
            outputs = ActivityGenerationModel.model.generate(**inputs, max_new_tokens = 300, use_cache = True)
            output_text = ActivityGenerationModel.tokenizer.batch_decode(outputs)[0].replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
            response_search = re.search("### Response:\n", output_text)
            response = output_text[response_search.start(): ].replace("### Response:\n", "")
            
            gc.collect()
            torch.cuda.empty_cache()
            return response
            
activity_generator = ActivityGenerator()