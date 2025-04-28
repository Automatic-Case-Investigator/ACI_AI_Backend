from query_generation_endpoint.objects.query_generation.query_generation_model import QueryGenerationModel
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from ACI_AI_Backend.objects.mutex_lock import lock
from huggingface_hub import hf_hub_download
from unsloth import is_bfloat16_supported
from django.conf import settings
from datasets import Dataset
from trl import SFTTrainer
import traceback
import torch
import json
import os
import re
import gc

class QueryGenerator:
    def __init__(self):
        file = open(settings.QUERY_GENERATION_CONFIG_PATH, "r")
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
    
    def generate_query(self, is_splunk, case_title, case_description, task_title, description, activity):
        with lock:
            self.cleanup()
            output_text = ""
            
            try:
                if QueryGenerationModel.model is None or QueryGenerationModel.tokenizer is None:
                    QueryGenerationModel.load()
                    
                prompt = ""
                if is_splunk:
                    return None
                else:
                    prompt = self.instruction["open_search"]

                input_string = f"Case title: {case_title}\nCase description: {case_description}\nTask title: {task_title}\nTask description: {description}\nActivity: {activity}"
                inputs = QueryGenerationModel.tokenizer(
                [
                    self.prompt_backbone.format(
                        prompt,
                        input_string,
                        "",
                    )
                ], return_tensors = "pt").to("cuda")
                
                outputs = QueryGenerationModel.model.generate(**inputs, max_new_tokens = 600)
                output_text = QueryGenerationModel.tokenizer.batch_decode(outputs)[0]         
            except:
                print(traceback.format_exc())
                raise OutOfMemoryError("Ran out of GPU VRAM for query generation. Please make sure that your GPU has enough vram for the model.")
                  
            response_tag = r"<\|start_header_id\|>assistant<\|end_header_id\|>"
            end_tag = r"<\|eot_id\|>|<\|start_header_id\|>system<\|end_header_id\|>|<\|start_header_id\|>user<\|end_header_id\|>"
            
            response_search = re.search(response_tag, output_text)
            response = output_text[response_search.start(): ]
            response = re.sub(response_tag, "", response)
            
            end_tag_search = re.search(end_tag, response)
            if end_tag_search is not None:
                response = response[ :end_tag_search.start()]
            
            QueryGenerationModel.unload()
            return response.strip()
            
query_generator = QueryGenerator()