from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
import json

from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher


class ActivityGenerationAgent(LLM):
    """Agent that generates activity descriptions.

    Parameters:
        deploy_method (str): Deployment method for the LLM.
        base_url (str): Base URL for the LLM service.
    """

    def __init__(self, deploy_method: str, base_url: str):
        """Load configuration and initialize the LLM.

        Parameters:
            deploy_method (str): Deployment method for the LLM.
            base_url (str): Base URL for the LLM service.

        Returns:
            None
        """
        with open(settings.ACTIVITY_GENERATION_CONFIG_PATH, "r") as file:
            config = json.load(file)

        self.prompt = config["instruction"]
        super().__init__(deploy_method=deploy_method, model_name=config["model_name"], base_url=base_url)

    def invoke(
        self,
        case_title: str,
        case_description: str,
        task_title: str,
        task_description: str,
        web_search_context: dict | None = None,
    ) -> str:
        """Generate an activity description based on case and task details.

        Parameters:
            case_title (str): Title of the case.
            case_description (str): Description of the case.
            task_title (str): Title of the task.
            task_description (str): Description of the task.
            web_search_context (dict or None): Optional web search context for background knowledge.

        Returns:
            str: The LLM's generated activity description.
        """
        messages = [
            ("system", self.prompt),
            ("human", f"Case title: {case_title}"),
            ("human", f"Case description: {case_description}"),
            ("human", f"Task title: {task_title}"),
            ("human", f"Task description: {task_description}"),
        ]

        if web_search_context:
            web_search_context_str = WebSearcher.context_to_str(web_search_context)
            messages.append(
                ("system", f"Here are the web search results for the relevant keywords. Treat these results as background knowledge only:\n{web_search_context_str}")
            )
        return super().invoke(messages)


activity_generation_agent = ActivityGenerationAgent(deploy_method=settings.DEPLOY_METHOD, base_url=settings.BASE_URL)
