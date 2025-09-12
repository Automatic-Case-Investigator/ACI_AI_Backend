from abc import ABC, abstractmethod

class OllamaAgent(ABC):
    def __init__(self, url):
        self.url = url
        
    def call_ollama(self, prompt: str) -> str:
        """
        Sends the given prompt to the Ollama model and returns the raw text response.
        """
        payload = {
            "model": "deepseek",
            "prompt": prompt,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.RequestException as e:
            raise RuntimeError(f"Error calling Ollama API: {e}") from e
        
    @abstractmethod
    def invoke(self, data: str) -> str:
        pass
    
    @abstractmethod
    def _get_prompt(self) -> str:
        pass

    @abstractmethod
    def _parse_response(self, response: str) -> str:
        pass
