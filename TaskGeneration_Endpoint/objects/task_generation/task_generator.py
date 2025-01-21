from TaskGeneration_Endpoint.objects.task_generation.task_generation_trainer import TaskGenerationTrainer
from TaskGeneration_Endpoint.objects.task_generation.task_generation_model import TaskGenerationModel
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from unsloth import is_bfloat16_supported
from django.conf import settings
from datasets import Dataset
from trl import SFTTrainer
import threading
import json
import os
import re

inference_lock = threading.Lock()

class TaskGenerator:
    def __init__(self):
        file = open(settings.TASK_GENERATION_CONFIG_PATH, "r")
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
    
    def generate_task(self, title, description):
        with inference_lock:
            if TaskGenerationModel.model is None or TaskGenerationModel.tokenizer is None:
                TaskGenerationModel.load()

            input_string = f"Title:{title}\n\nDescription:{description}"
            inputs = TaskGenerationModel.tokenizer(
            [
                self.prompt_backbone.format(
                    self.instruction,
                    input_string,
                    "",
                )
            ], return_tensors = "pt").to("cuda")
            
            outputs = TaskGenerationModel.model.generate(**inputs, max_new_tokens = 300, use_cache = True)
            output_text = TaskGenerationModel.tokenizer.batch_decode(outputs)[0].replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
            response_search = re.search("### Response:\n", output_text)
            response = output_text[response_search.start(): ].replace("### Response:\n", "")
            return response
            
task_generator = TaskGenerator()