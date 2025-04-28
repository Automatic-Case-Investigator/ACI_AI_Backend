from activity_generation_endpoint.objects.activity_generation.activity_generation_model import ActivityGenerationModel
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from ACI_AI_Backend.objects.redis_client import redis_client
from ACI_AI_Backend.objects.mutex_lock import lock
from huggingface_hub import snapshot_download
from unsloth import is_bfloat16_supported
from datasets import Dataset
from django.conf import settings
from trl import SFTTrainer
from hashlib import sha256
import shutil
import torch
import json
import time
import os
import gc


class ActivityGenerationTrainer:
    def __init__(self):
        file = open(settings.ACTIVITY_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)
        
        self.repo_name = config["repo_name"]
        self.model_name = config["model_name"]
        self.dataset_key_prefix = config["dataset_key_prefix"]
        self.workspace_dir = config["workspace_dir"]
        self.local_model_dir = config["local_model_dir"]
        self.instruction = config["instruction"]
        self.max_seq_length = config["max_seq_length"]
        self.load_in_4bit = config["load_in_4bit"]
        self.dtype = config["dtype"]
        file.close()
        
        self.prompt_backbone = """
        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
        
        self.dataset = None
    
    def formatting_prompts_func(self, examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = self.prompt_backbone.format(instruction, input, output)
            texts.append(text)
        return {
            "text": texts,
        }
    
    def load_baseline(self):
        if os.path.exists(self.local_model_dir) and os.path.isdir(self.local_model_dir):
            # delete all the old files in the model directory
            os.system(f"rm -rfd {self.local_model_dir}*")

        os.system(f"mkdir -p {self.local_model_dir}")

        snapshot_download(repo_id=self.repo_name, local_dir=self.local_model_dir)

    def backup_model(self):
        current_timestamp = time.time()
        zip_name = sha256(str(current_timestamp).encode('utf-8')).hexdigest()
        shutil.make_archive(self.workspace_dir + zip_name, 'zip', self.local_model_dir)
        return self.workspace_dir + zip_name + ".zip"

    def load_model_tokenizer_locally(self):
        ActivityGenerationModel.load()

    def load_dataset(self):
        dataset_dict = {"instruction": [], "input": [], "output": []}

        for key in redis_client.scan_iter(match=self.dataset_key_prefix):
            task_data = json.loads(redis_client.get(key))
            input_data = f'Title:\n{task_data["title"]}\nDescription:\n{task_data["description"]}'.replace("\r", "")
            output_data = ""
            for index in range(len(task_data["activities"])):
                output_data += f'{index + 1}. {task_data["activities"][index]["title"]}\n'.replace("\r", "")

            dataset_dict["instruction"].append(self.instruction)
            dataset_dict["input"].append(input_data)
            dataset_dict["output"].append(output_data)
        
        if (len(dataset_dict["instruction"]) == 0 or len(dataset_dict["input"]) == 0 or len(dataset_dict["output"]) == 0):
            raise ValueError("No dataset is loaded")

        self.dataset = Dataset.from_dict(dataset_dict)
        self.dataset = self.dataset.map(self.formatting_prompts_func, batched=True)
        
        # Delete all used dataset
        for key in redis_client.scan_iter(self.dataset_key_prefix):
            redis_client.delete(key)

    def train(self, seed=3407, max_steps=200, learning_rate=2e-4, gradient_accumulation_steps=4, weight_decay=0.00001):
        with lock:
            trainer = SFTTrainer(
                model=ActivityGenerationModel.model,
                tokenizer=ActivityGenerationModel.tokenizer,
                train_dataset=self.dataset,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                dataset_num_proc=2,
                packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=5,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=weight_decay,
                    lr_scheduler_type="linear",
                    seed=seed,
                    output_dir="outputs",
                ),
            )

            trainer.train()

            ActivityGenerationModel.model.save_pretrained(self.local_model_dir)
            ActivityGenerationModel.tokenizer.save_pretrained(self.local_model_dir)
            
            del trainer
            gc.collect()
            torch.cuda.empty_cache()