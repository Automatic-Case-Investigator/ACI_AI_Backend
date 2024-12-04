from TaskGeneration_Endpoint.objects.task_generation.task_generation_trainer import TaskGenerationTrainer
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
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
        
        self.repo_name = config["repo_name"]
        self.local_model_dir = config["local_model_dir"]
        self.instruction = config["instruction"]
        self.max_seq_length = config["max_seq_length"]
        self.load_in_4bit = config["load_in_4bit"]
        self.dtype = config["dtype"]
        self.prompt_backbone = """
        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
        file.close()
        
        try:
            if not os.path.exists(self.local_model_dir) or not os.path.isdir(self.local_model_dir):

                # pull the model from remote repository if no model is saved locally
                os.system(f"mkdir -p {self.local_model_dir}")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    self.repo_name,
                    max_seq_length=self.max_seq_length,
                    dtype = self.dtype,
                    load_in_4bit = self.load_in_4bit,
                )
                self.model.save_pretrained(self.local_model_dir)
                self.tokenizer.save_pretrained(self.local_model_dir)
            else:
                
                # load the saved model
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    self.local_model_dir,
                    max_seq_length=self.max_seq_length,
                    dtype = self.dtype,
                    load_in_4bit = self.load_in_4bit,
                )
            FastLanguageModel.for_inference(self.model)
        except Exception as e:
            print(e)
    
    def generate_task(self, title, description):
        with inference_lock:
            input_string = f"Title:{title}\n\nDescription:{description}"
            inputs = self.tokenizer(
            [
                self.prompt_backbone.format(
                    self.instruction,
                    input_string,
                    "",
                )
            ], return_tensors = "pt").to("cuda")
            
            outputs = self.model.generate(**inputs, max_new_tokens = 300, use_cache = True)
            output_text = self.tokenizer.batch_decode(outputs)[0].replace("<|begin_of_text|>", "").replace("<|eot_id|>", "")
            response_search = re.search("### Response:\n", output_text)
            response = output_text[response_search.start(): ].replace("### Response:\n", "")
            return response
            
task_generator = TaskGenerator()