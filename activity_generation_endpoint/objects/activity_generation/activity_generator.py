from activity_generation_endpoint.objects.activity_generation.activity_generation_model import (
    ActivityGenerationModel,
)
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from ACI_AI_Backend.objects.string_text_streamer import StringTextStreamer
from ACI_AI_Backend.objects.mutex_lock import lock
from django.conf import settings
import traceback
import torch
import json
import gc


class ActivityGenerator:
    def __init__(self):
        file = open(settings.ACTIVITY_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)
        self.instruction = config["instruction"]
        file.close()

    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()

    def generate_activity(self, case_title, case_description, task_title, description):
        with lock:
            self.cleanup()
            output_text = ""

            try:
                # Loads the activity generation model and tokenizer if not loaded
                if (
                    ActivityGenerationModel.model is None
                    or ActivityGenerationModel.tokenizer is None
                ):
                    ActivityGenerationModel.load()

                input_string = f"Case title: {case_title}\nCase description: {case_description}\nTask title: {task_title}\nTask description: {description}"
                messages = [
                    {
                        "role": "user",
                        "content": f"{self.instruction}\nYour input is:\n{input_string}",
                    }
                ]

                # Formats and tokenizes the input string
                input_text = ActivityGenerationModel.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, padding=True
                )
                input_ids = ActivityGenerationModel.tokenizer(
                    [input_text], return_tensors="pt"
                ).to("cuda")

                # Generates output from tokenized input
                streamer = StringTextStreamer(
                    ActivityGenerationModel.tokenizer, skip_prompt=True
                )
                _ = ActivityGenerationModel.model.generate(
                    **input_ids,
                    streamer=streamer,
                    max_new_tokens=600,
                    use_cache=True,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                )
                output_text = streamer.get_output()

            except:
                print(traceback.format_exc())
                raise OutOfMemoryError(
                    "Ran out of GPU VRAM for activity generation. Please make sure that your GPU has enough vram for the model."
                )

            ActivityGenerationModel.unload()
            return output_text


activity_generator = ActivityGenerator()
