from transformers import TextStreamer
import io

class StringTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.output_buffer = io.StringIO()

    def on_finalized_text(self, text, stream_end=False):
        self.output_buffer.write(text)

    def get_output(self):
        return self.output_buffer.getvalue()