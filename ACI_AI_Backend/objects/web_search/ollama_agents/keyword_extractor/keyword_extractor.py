from ACI_AI_Backend.objects.web_search.ollama_agents.ollama_agent import OllamaAgent
import json
import re

class KeywordExtractor(OllamaAgent):
    def __init__(self, url):
        super().__init__(url)
        
        prompt_path = "./ACI_AI_Backend/objects/web_search/ollama_agents/keyword_extractor/prompt/prompt.json"
        
        with open(prompt_path, "r") as file:
            self.prompt = json.load(file)
        
    def invoke(self, data: str) -> str:
        full_prompt = self._get_prompt() + data
        raw_response = self.call_ollama(full_prompt)
        return self._parse_response(raw_response)
    
    def _get_prompt(self) -> str:
        output = ""
        output += self.prompt["system_role"]
        output += self.prompt["task"] + "\n\n"
        
        output += "Instructions:\n"
        for instruction in self.prompt["instructions"]:
            output += instruction + "\n"
        output += "\n"
            
        output += "Input Format:\n"
        for input_format in self.prompt["input_format"]:
            output += input_format + "\n"
        output += "\n"
            
        output += "Output Format:\n"
        for output_format in self.prompt["output_format"]:
            output += output_format + "\n"
        output += "\n"
            
        output += "Example Input:\n"
        for input_example in self.prompt["input_example"]:
            output += input_example + "\n"
        output += "\n"
            
        output += "Example Output:\n"
        for output_example in self.prompt["output_example"]:
            output += output_example + "\n"
        output += "\n"
        
        output += "Below is the input you will work with:\n"
        return output

    def _parse_response(self, response: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return cleaned
