import json
from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher

with open(settings.TASK_GENERATION_CONFIG_PATH, "r") as file:
    _CONFIG = json.load(file)


class TaskGenerationAgent(LLM):
    """Agent that generates tasks based on case information.

    Parameters:
        deploy_method (str): Deployment method for the LLM.
        base_url (str): Base URL for the LLM service.
    """

    def __init__(self, deploy_method: str, base_url: str):
        """Initialize the TaskGenerationAgent with the prompt from configuration.

        Parameters:
            deploy_method (str): Deployment method for the LLM.
            base_url (str): Base URL for the LLM service.

        Returns:
            None
        """
        self.prompt = _CONFIG["instruction"]
        super().__init__(deploy_method=deploy_method, model_name=_CONFIG["model_name"], base_url=base_url)

    def invoke(self, case_title: str, case_description: str, web_search_context: dict | None = None) -> str:
        """Generate a task based on case title, description, and optional web search context.

        Parameters:
            case_title (str): Title of the case.
            case_description (str): Description of the case.
            web_search_context (dict or None): Optional web search context providing background knowledge.

        Returns:
            str: The LLM's generated task description.
        """
        messages = [
            ("system", self.prompt),
            ("human", f"Case title: {case_title}"),
            ("human", f"Case description: {case_description}"),
        ]

        if web_search_context:
            web_search_context_str = WebSearcher.context_to_str(web_search_context)
            messages.append(("system", f"Here are the web search results for the relevant keywords. Treat these results as background knowledge only:\n{web_search_context_str}"))

        return super().invoke(messages)


task_generation_agent = TaskGenerationAgent(deploy_method=settings.DEPLOY_METHOD, base_url=settings.BASE_URL)
