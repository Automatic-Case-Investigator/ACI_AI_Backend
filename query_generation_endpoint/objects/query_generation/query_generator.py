from query_generation_endpoint.objects.query_generation.query_generation_model import (
    QueryGenerationModel,
)
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from ACI_AI_Backend.objects.string_text_streamer import StringTextStreamer
from ACI_AI_Backend.objects.mutex_lock import lock
from django.conf import settings
import traceback
import torch
import json
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

    def generate_query(
        self, is_splunk, case_title, case_description, task_title, description, activity
    ):
        with lock:
            self.cleanup()
            output_text = ""

            try:
                # Loads the query generation model and tokenizer if not loaded
                if (
                    QueryGenerationModel.model is None
                    or QueryGenerationModel.tokenizer is None
                ):
                    QueryGenerationModel.load()

                # Decides which instruction prompt to use depending on SIEM architecture
                instruction = ""
                if is_splunk:
                    return None
                else:
                    instruction = self.instruction["open_search"]

                input_string = f"Case title: {case_title}\nCase description: {case_description}\nTask title: {task_title}\nTask description: {description}\nActivity: {activity}"
                messages = [
                    {
                        "role": "user",
                        "content": f"{instruction}\nYour input is:\n{input_string}",
                    }
                ]

                # Formats and tokenizes the input string
                input_text = QueryGenerationModel.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, padding=True
                )
                input_ids = QueryGenerationModel.tokenizer(
                    [input_text], return_tensors="pt"
                ).to("cuda")

                # Generates output from tokenized input
                streamer = StringTextStreamer(
                    QueryGenerationModel.tokenizer, skip_prompt=True
                )
                _ = QueryGenerationModel.model.generate(
                    **input_ids,
                    streamer=streamer,
                    max_new_tokens=600,
                    use_cache=True,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.2,
                    repetition_penalty=1.2
                )
                output_text = streamer.get_output()

            except:
                print(traceback.format_exc())
                raise OutOfMemoryError(
                    "Ran out of GPU VRAM for query generation. Please make sure that your GPU has enough vram for the model."
                )

            QueryGenerationModel.unload()
            return output_text


query_generator = QueryGenerator()
