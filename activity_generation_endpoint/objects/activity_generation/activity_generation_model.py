from unsloth import FastLanguageModel
from django.conf import settings
import torch
import json
import os

class ActivityGenerationModel:
    model = None
    tokenizer = None

    @classmethod
    def load(self):
        file = open(settings.ACTIVITY_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)
        
        repo_name = config["repo_name"]
        local_model_dir = config["local_model_dir"]
        instruction = config["instruction"]
        max_seq_length = config["max_seq_length"]
        load_in_4bit = config["load_in_4bit"]
        dtype = config["dtype"]
        file.close()
        
        try:
            if not os.path.exists(local_model_dir) or not os.path.isdir(local_model_dir):

                # pull the model from remote repository if no model is saved locally
                os.system(f"mkdir -p {local_model_dir}")
                ActivityGenerationModel.model, ActivityGenerationModel.tokenizer = FastLanguageModel.from_pretrained(
                    repo_name,
                    max_seq_length=max_seq_length,
                    dtype = dtype,
                    load_in_4bit = load_in_4bit,
                )
                ActivityGenerationModel.model.save_pretrained(local_model_dir)
                ActivityGenerationModel.tokenizer.save_pretrained(local_model_dir)
            else:
                
                # load the saved model
                ActivityGenerationModel.model, ActivityGenerationModel.tokenizer = FastLanguageModel.from_pretrained(
                    local_model_dir,
                    max_seq_length=max_seq_length,
                    dtype = dtype,
                    load_in_4bit = load_in_4bit,
                )
            FastLanguageModel.for_inference(ActivityGenerationModel.model)
        except Exception as e:
            print(e)

    @classmethod
    def unload(self):
        del ActivityGenerationModel.model
        del ActivityGenerationModel.tokenizer
        
        ActivityGenerationModel.model = None
        ActivityGenerationModel.tokenizer = None
        
        torch.cuda.empty_cache()