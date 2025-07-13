from task_generation_endpoint.objects.task_generation.task_generation_model import (
    TaskGenerationModel,
)
from ACI_AI_Backend.objects.redis_client import redis_client
from ACI_AI_Backend.objects.mutex_lock import lock
from huggingface_hub import snapshot_download
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from django.conf import settings
from unsloth import to_sharegpt
from datasets import Dataset
from hashlib import sha256
from trl import SFTTrainer
import shutil
import torch
import json
import time
import os
import gc


class TaskGenerationTrainer:
    def __init__(self):
        file = open(settings.TASK_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)

        self.dataset_key_prefix = config["dataset_key_prefix"]
        self.workspace_dir = config["workspace_dir"]
        self.local_model_dir = config["local_model_dir"]
        self.instruction = config["instruction"]
        self.max_seq_length = config["max_seq_length"]
        self.load_in_4bit = config["load_in_4bit"]
        self.dtype = config["dtype"]
        file.close()

        self.dataset = None

    def load_baseline(self, model_id):
        file = open(settings.TASK_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)
        self.repo_name = config["models"][model_id]["repo_name"]
        file.close()

        if os.path.exists(self.local_model_dir) and os.path.isdir(self.local_model_dir):
            # delete all the old files in the model directory
            os.system(f"rm -rfd {self.local_model_dir}*")

        os.system(f"mkdir -p {self.local_model_dir}")
        snapshot_download(
            repo_id=self.repo_name,
            local_dir=self.local_model_dir,
            max_workers=16,
            resume_download=True,
        )

    def backup_model(self):
        current_timestamp = time.time()
        zip_name = sha256(str(current_timestamp).encode("utf-8")).hexdigest()
        shutil.make_archive(self.workspace_dir + zip_name, "zip", self.local_model_dir)
        return self.workspace_dir + zip_name + ".zip"

    def load_model_tokenizer_locally(self):
        TaskGenerationModel.load()

    def load_dataset(self):
        dataset_dict = {"instruction": [], "input": [], "output": []}

        # Loads the temporary data entry stored in the redis database
        for key in redis_client.scan_iter(match=self.dataset_key_prefix):
            case_data = json.loads(redis_client.get(key))
            input_data = f'Title:\n{case_data["title"]}\n\nDescription:\n{case_data["description"]}'.replace(
                "\r", ""
            )
            output_data = ""
            for index in range(len(case_data["tasks"])):
                output_data += f'Task #{index + 1}\nTitle: {case_data["tasks"][index]["title"]}\nDescription: {case_data["tasks"][index]["description"]}\n\n'.replace(
                    "\r", ""
                )

            dataset_dict["instruction"].append(self.instruction)
            dataset_dict["input"].append(input_data)
            dataset_dict["output"].append(output_data)

        if (
            len(dataset_dict["instruction"]) == 0
            or len(dataset_dict["input"]) == 0
            or len(dataset_dict["output"]) == 0
        ):
            raise ValueError("No dataset is loaded")

        # Creates a dataset object from dictionary
        self.dataset = Dataset.from_dict(dataset_dict)

        # Standardizes the dataset to sharegpt format
        self.dataset = standardize_sharegpt(to_sharegpt(
            self.dataset,
            merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
            output_column_name="output",
            conversation_extension=1,
        ))

        try:
            # Applies the model template
            self.dataset = apply_chat_template(
                self.dataset,
                tokenizer=TaskGenerationModel.tokenizer
            )
        except:
            raise RuntimeError("Tokenizer is not initialized")

        # Delete all used dataset
        for key in redis_client.scan_iter(self.dataset_key_prefix):
            redis_client.delete(key)

    def train(
        self,
        seed=3407,
        max_steps=200,
        learning_rate=4e-4,
        gradient_accumulation_steps=4,
        weight_decay=0.001,
    ):
        with lock:
            trainer = SFTTrainer(
                model=TaskGenerationModel.model,
                tokenizer=TaskGenerationModel.tokenizer,
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

            TaskGenerationModel.model.save_pretrained(self.local_model_dir)
            TaskGenerationModel.tokenizer.save_pretrained(self.local_model_dir)

            del trainer
            gc.collect()
            torch.cuda.empty_cache()
