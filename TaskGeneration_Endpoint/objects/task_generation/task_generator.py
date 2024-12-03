from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
import json
import os
import re

class TaskGenerator:
    def __init__(self):
        self.local_model_dir = "./models/task_generation/model/"
        self.max_seq_length = 2048
        self.load_in_4bit = True
        self.dtype = None
        self.prompt_backbone = """
        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
        self.instruction = "You are a soc analyst. You received a case in the soar platform, including detailed information about an alert. The title section includes a brief description of the case, and the description section includes detailed information about the case. Based on the case information, list only the tasks you would suggest to create for investigating the incident. For each task, write only one sentence for title and description. Your answer should follow this format:Task # Title: <title> Description: <description>... Here is the decoded data of the case:"
        
        try:
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