from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
import json


class ActivityGenerationAgent(LLM):
    def __init__(self, deploy_method: str, base_url: str):
        with open(settings.ACTIVITY_GENERATION_CONFIG_PATH, "r") as file:
            config = json.load(file)

        self.prompt = config["instruction"]
        super().__init__(deploy_method=deploy_method, model_name=config["model_name"], base_url=base_url)
    
    def invoke(self, case_title: str, case_description: str, task_title: str, task_description: str, web_search_context: dict | None = None) -> str:
        messages = [
            ("system", self.prompt),
            ("human", f"Case title: {case_title}"),
            ("human", f"Case description: {case_description}"),
            ("human", f"Task title: {task_title}"),
            ("human", f"Task description: {task_description}")
        ]

        if web_search_context:
            web_search_context_str = ""
            for keyword in web_search_context.keys():
                explaination = web_search_context[keyword]["explanation"]
                if len(explaination) == 0:
                    continue

                web_search_context_str += f"{keyword}:\n```\n{explaination}\n```\n\n"

            messages.append(("system", f"Here are the web search results for the relevant keywords. Treat these results as background knowledge only. When generating tasks, use the actual case title and description:\n{web_search_context_str}"))

        return super().invoke(messages)


activity_generation_agent = ActivityGenerationAgent(deploy_method=settings.DEPLOY_METHOD, base_url=settings.BASE_URL)
