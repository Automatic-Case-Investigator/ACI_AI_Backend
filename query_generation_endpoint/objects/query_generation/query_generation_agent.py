from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
import json

from ACI_AI_Backend.objects.prompt_injection_detection.prompt_injection_detector import PromptInjectionDetector


class QueryGenerationAgent(LLM):
    def __init__(self, deploy_method: str, base_url: str):
        self.rejection_message = "Could not provide answer to the given instructions"
        
        with open(settings.QUERY_GENERATION_CONFIG_PATH, "r") as file:
            self.config = json.load(file)

        super().__init__(deploy_method=deploy_method, model_name=self.config["model_name"], base_url=base_url)

    def invoke(
        self,
        is_splunk: bool,
        user_prompt: str | None = None,
        case_title: str | None = None,
        case_description: str | None = None,
        task_title: str | None = None,
        task_description: str | None = None,
        activity: str | None = None,
        web_search_context: dict | None = None,
    ):
        messages = []

        if is_splunk:
            # To be implemented
            pass

        else:
            if user_prompt:
                injection_detector = PromptInjectionDetector()
                if not injection_detector.is_safe(user_prompt):
                    return self.rejection_message

                prompt = self.config["instruction"]["open_search"]["from_prompt"]
                messages = [("system", prompt), ("human", user_prompt)]

            elif all([case_title, case_description, task_title, task_description, activity]):
                prompt = self.config["instruction"]["open_search"]["from_case"]
                messages = [
                    ("system", prompt),
                    ("human", f"Case title: {case_title}"),
                    ("human", f"Case description: {case_description}"),
                    ("human", f"Task title: {task_title}"),
                    ("human", f"Task description: {task_description}"),
                    ("human", f"Activity: {activity}"),
                ]
            else:
                raise ValueError("Invalid arguments for invoke()")

        if web_search_context:
            web_search_context_str = ""
            for keyword, explaination in web_search_context.items():
                if len(explaination) == 0:
                    continue

                web_search_context_str += f"{keyword}:\n```\n{explaination}\n```\n\n"

            messages.append(
                (
                    "system",
                    f"Here are the web search results for the relevant keywords. Treat these results as background knowledge only. When generating tasks, use the actual case title and description:\n{web_search_context_str}",
                )
            )

        return super().invoke(messages)


query_generation_agent = QueryGenerationAgent(deploy_method=settings.DEPLOY_METHOD, base_url=settings.BASE_URL)
