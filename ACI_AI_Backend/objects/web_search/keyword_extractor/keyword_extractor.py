from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
import json

class KeywordExtractor(LLM):
    def __init__(self):
        prompt_path = settings.KEYWORD_EXTRACTOR_CONFIG_PATH
        with open(prompt_path, "r") as file:
            config = json.load(file)
        
        self.prompt = config["instruction"]
        super().__init__(
            deploy_method=settings.DEPLOY_METHOD,
            model_name=config["model_name"],
            base_url=settings.BASE_URL,
        )

    def invoke(self, text: str) -> str:
        messages = [
            ("system", self.prompt),
            ("human", text)
        ]

        return super().invoke(messages)
