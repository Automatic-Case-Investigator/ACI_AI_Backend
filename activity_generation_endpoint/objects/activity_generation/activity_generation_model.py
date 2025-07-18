from unsloth import FastLanguageModel
from django.conf import settings
import torch
import json
import os
import gc


class ActivityGenerationModel:
    model = None
    tokenizer = None

    @classmethod
    def load(self):
        file = open(settings.ACTIVITY_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)

        # TODO: change the current model id to current_backup_model.model_id when implementing backup functionality
        repo_name = config["models"]["activity_gen_llama_3b"]["repo_name"]
        local_model_dir = config["local_model_dir"]
        instruction = config["instruction"]
        max_seq_length = config["max_seq_length"]
        load_in_4bit = config["load_in_4bit"]
        dtype = config["dtype"]
        file.close()

        if not os.path.exists(local_model_dir) or not os.path.isdir(local_model_dir):
            # pull the model from remote repository if no model is saved locally
            os.system(f"mkdir -p {local_model_dir}")
            ActivityGenerationModel.model, ActivityGenerationModel.tokenizer = (
                FastLanguageModel.from_pretrained(
                    repo_name,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                    device_map="auto",
                    fix_tokenizer=False,
                )
            )
            ActivityGenerationModel.model.save_pretrained(local_model_dir)
            ActivityGenerationModel.tokenizer.save_pretrained(local_model_dir)
        else:
            # load the saved model
            ActivityGenerationModel.model, ActivityGenerationModel.tokenizer = (
                FastLanguageModel.from_pretrained(
                    local_model_dir,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                    device_map="auto",
                    fix_tokenizer=False,
                )
            )
        FastLanguageModel.for_inference(ActivityGenerationModel.model)

    @classmethod
    def unload(self):
        try:
            del ActivityGenerationModel.model
            del ActivityGenerationModel.tokenizer
            ActivityGenerationModel.model = None
            ActivityGenerationModel.tokenizer = None

            gc.collect()
            torch.cuda.empty_cache()
        except:
            print("Model already unloaded")
