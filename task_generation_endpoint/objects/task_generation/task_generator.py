from task_generation_endpoint.objects.task_generation.task_generation_trainer import TaskGenerationTrainer
from task_generation_endpoint.objects.task_generation.task_generation_model import TaskGenerationModel
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


class TaskGenerator:
    def __init__(self):
        file = open(settings.TASK_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)

        self.instruction = config["instruction"]
        self.prompt_backbone = """
<|start_header_id|>system<|end_header_id|>
{}<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}
"""
        file.close()
    
    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()
            
    def generate_task(self, title, description):
        with lock:
            self.cleanup()
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
            
            response_tag = r"<\|start_header_id\|>assistant<\|end_header_id\|>"
            end_tag = r"<\|eot_id\|>|<\|start_header_id\|>system<\|end_header_id\|>|<\|start_header_id\|>user<\|end_header_id\|>"

            outputs = TaskGenerationModel.model.generate(**inputs, max_new_tokens = 300, use_cache = True)
            output_text = TaskGenerationModel.tokenizer.batch_decode(outputs)[0].replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
            
            response_search = re.search(response_tag, output_text)
            response = output_text[response_search.start(): ]
            response = re.sub(response_tag, "", response)
            
            end_tag_search = re.search(end_tag, response)
            if end_tag_search is not None:
                response = response[ :end_tag_search.start()]
            
            self.cleanup()
            return response.strip()
            
task_generator = TaskGenerator()