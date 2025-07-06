from query_generation_endpoint.objects.query_generation.query_generation_model import QueryGenerationModel
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from ACI_AI_Backend.objects.string_text_streamer import StringTextStreamer
from ACI_AI_Backend.objects.mutex_lock import lock
from huggingface_hub import hf_hub_download
from unsloth import is_bfloat16_supported
from transformers import TextStreamer
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
        file.close()
    
    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()
    
    def generate_query(self, is_splunk, case_title, case_description, task_title, description, activity):
        with lock:
            self.cleanup()
            output_text = ""
            
            try:
                # Loads the query generation model and tokenizer if not loaded
                if QueryGenerationModel.model is None or QueryGenerationModel.tokenizer is None:
                    QueryGenerationModel.load()
                
                # Decides which instruction prompt to use depending on SIEM architecture
                instruction = ""
                if is_splunk:
                    return None
                else:
                    instruction = self.instruction["open_search"]

                input_string = f"Case title: {case_title}\nCase description: {case_description}\nTask title: {task_title}\nTask description: {description}\nActivity: {activity}"
                messages = [
                    {"role": "user", "content" : f"{instruction}\nYour input is:\n{input_string}"}
                ]
                
                # Formats and tokenizes the input string
                input_text = QueryGenerationModel.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, padding=True)
                input_ids = QueryGenerationModel.tokenizer([input_text], return_tensors = "pt").to("cuda")

                # Generates output from tokenized input
                streamer = StringTextStreamer(QueryGenerationModel.tokenizer, skip_prompt=True)
                _ = QueryGenerationModel.model.generate(**input_ids, streamer=streamer, max_new_tokens=600, use_cache=True)
                output_text = streamer.get_output()        
            
            except:
                print(traceback.format_exc())
                raise OutOfMemoryError("Ran out of GPU VRAM for query generation. Please make sure that your GPU has enough vram for the model.")

            
            QueryGenerationModel.unload()
            return output_text
            
query_generator = QueryGenerator()