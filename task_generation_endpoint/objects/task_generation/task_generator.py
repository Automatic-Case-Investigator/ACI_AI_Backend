from task_generation_endpoint.objects.task_generation.task_generation_model import (
    TaskGenerationModel,
)
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from ACI_AI_Backend.objects.string_text_streamer import StringTextStreamer
from ACI_AI_Backend.objects.mutex_lock import lock
from django.conf import settings
import traceback
import torch
import json
import gc


class TaskGenerator:
    def __init__(self):
        file = open(settings.TASK_GENERATION_CONFIG_PATH, "r")
        config = json.load(file)
        self.instruction = config["instruction"]
        self.web_search_prompt = config["web_search_prompt"]
        file.close()

    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()

    def generate_task(self, title: str, description: str, context: dict = None) -> str:
        with lock:
            self.cleanup()
            output_text = ""

            try:
                # Loads the task generation model and tokenizer if not loaded
                if (
                    TaskGenerationModel.model is None
                    or TaskGenerationModel.tokenizer is None
                ):
                    TaskGenerationModel.load()

                input_string = f"Title:{title}\n\nDescription:{description}"
                messages = []

                if context:
                    context_str = ""
                    for keyword, explaination in context.items():
                        if len(explaination) == 0:
                            continue

                        context_str += f"{keyword}:\n```\n{explaination}\n```\n\n"

                    messages.append(
                        {
                            "role": "user",
                            "content": f"{self.web_search_prompt}\n{context_str}",
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": f"{self.instruction}\nYour input is:\n{input_string}",
                    }
                )

                # Formats and tokenizes the input string
                input_text = TaskGenerationModel.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, padding=True
                )
                input_ids = TaskGenerationModel.tokenizer(
                    [input_text], return_tensors="pt"
                ).to("cuda")

                # Generates output from tokenized input
                streamer = StringTextStreamer(
                    TaskGenerationModel.tokenizer, skip_prompt=True
                )
                _ = TaskGenerationModel.model.generate(
                    **input_ids,
                    streamer=streamer,
                    max_new_tokens=600,
                    use_cache=True,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.5,
                    repetition_penalty=1.1,
                )
                output_text = streamer.get_output()

            except:
                print(traceback.format_exc())
                raise OutOfMemoryError(
                    "Ran out of GPU VRAM for task generation. Please make sure that your GPU has enough vram for the model."
                )

            TaskGenerationModel.unload()
            return output_text


task_generator = TaskGenerator()
