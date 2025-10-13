from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from ACI_AI_Backend.objects.device import get_freest_device
from enum import Enum

class PromptSafetyLabel(Enum):
    SAFE = "SAFE"
    INJECTION = "INJECTION"

class PromptInjectionDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
        device = get_freest_device()
        
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=1024,
            device=device,
        )
        
    def is_safe(self, user_prompt: str) -> bool:
        result = self.classifier(user_prompt)
        if result[0]["label"] == PromptSafetyLabel.INJECTION.value:
            return False
        
        return True
    
    