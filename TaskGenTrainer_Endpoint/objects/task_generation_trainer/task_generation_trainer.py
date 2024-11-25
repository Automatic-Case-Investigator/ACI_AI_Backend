from ACI_AI_Backend.objects.model_update_notifier.model_update_notifier import ModelUpdateNotifier
from ACI_AI_Backend.objects.redis_client import redis_client
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
import json
import os


class TaskGenerationTrainer:
    def __init__(self):
        self.model_name = "unsloth/Llama-3.2-3B-Instruct"
        self.dataset_key_prefix = "Case:*"
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
        self.instruction = "You are a soc analyst. You received a case in the soar platform, including detailed information about an alert. Based on the case information, list only the tasks you would suggest to create for investigating the incident. For each task, write only one sentence for title and description. Your answer should follow this format:Task # Title: <title> Description: <description>... Here is the decoded data of the case:"

        self.model = None
        self.tokenizer = None
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

    def load_model_and_tokenizer(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    def load_dataset(self):
        dataset_dict = {"instruction": [], "input": [], "output": []}

        for key in redis_client.scan_iter(match=self.dataset_key_prefix):
            case_data = json.loads(redis_client.get(key))
            input_data = f"Title:\n{case_data["title"]}\n\nDescription:\n{case_data["description"]}"
            output_data = ""
            for index in range(len(case_data["tasks"])):
                output_data += f"Task #{index + 1}\nTitle: {case_data["tasks"][index]["title"]}\nDescription: {case_data["tasks"][index]["description"]}\n\n"

            dataset_dict["instruction"].append(self.instruction)
            dataset_dict["input"].append(input_data)
            dataset_dict["output"].append(output_data)
        
        if (len(dataset_dict["instruction"]) == 0 or len(dataset_dict["input"]) == 0 or len(dataset_dict["output"]) == 0):
            raise ValueError("No dataset is loaded")

        self.dataset = Dataset.from_dict(dataset_dict)
        self.dataset = self.dataset.map(self.formatting_prompts_func, batched=True)

    def train(self):
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=250,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.00001,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
            ),
        )

        trainer.train()

        self.model.save_pretrained("lora_model")
        self.tokenizer.save_pretrained("lora_model")
        self.model.save_pretrained_gguf(
            "model", self.tokenizer, quantization_method=["q8_0"]
        )

        # Delete all used dataset
        for key in redis_client.scan_iter(self.dataset_key_prefix):
            redis_client.delete(key)

        os.system("mv model/unsloth.Q8_0.gguf models/task_generation.gguf")
        ModelUpdateNotifier.notifyUpdate("task_generation")