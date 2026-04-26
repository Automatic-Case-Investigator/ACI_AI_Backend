from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher
from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
import json


class RelevencyFilterAgent(LLM):
    """
    Agent for filtering relevancy of SIEM queries.

    Parameters:
        deploy_method (str): Deployment method for the LLM.
        base_url (str): Base URL for the LLM service.

    Returns:
        None
    """

    def __init__(self, deploy_method: str, base_url: str):
        """
        Initialize the RelevencyFilterAgent with configuration from file.

        Parameters:
            deploy_method (str): Deployment method for the LLM.
            base_url (str): Base URL for the LLM service.

        Returns:
            None
        """
        with open(settings.RELEVENCY_FILTER_CONFIG_PATH, "r") as file:
            config = json.load(file)

        self.prompt = config["instruction"]
        super().__init__(deploy_method=deploy_method, model_name=config["model_name"], base_url=base_url, reasoning_effort="low")

    def invoke(
        self,
        siem: str = "wazuh",
        query: str = "",
        event: str = "",
        case_title: str = "",
        case_description: str = "",
        task_title: str = "",
        task_description: str = "",
        activity: str = "",
        additional_notes: str | None = None,
        web_search_context: dict | None = None,
    ) -> str:
        """
        Invoke the relevancy filtering process.

        Parameters:
            siem (str): SIEM type (default "wazuh").
            query (str): SIEM query string.
            event (str): Event details.
            case_title (str): Title of the case.
            case_description (str): Description of the case.
            task_title (str): Title of the task.
            task_description (str): Description of the task.
            activity (str): Activity description.
            additional_notes (str | None): Additional notes from the human SOC analyst expert for the investigation.
            web_search_context (dict or None): Context from web search.

        Returns:
            str: Result from the LLM invocation.
        """
        messages = []

        if siem == "splunk":
            # To be implemented
            pass

        else:
            messages = [
                ("system", self.prompt),
                ("human", f"# Case title:\n{case_title}\n---"),
                ("human", f"# Case description:\n{case_description}\n---"),
                ("human", f"# Task title:\n{task_title}\n---"),
                ("human", f"# Task description:\n{task_description}\n---"),
                ("human", f"# Activity:\n{activity}\n---"),
                ("human", f"# SIEM Query:\n{query}\n---"),
                ("human", f"# Event queried:\n{event}\n---"),
            ]

        if additional_notes is not None:
            messages.append(("human", f"# Additional notes from the human SOC analyst expert for the investigation.\n{additional_notes}\n---"))

        if web_search_context:
            web_search_context_str = WebSearcher.context_to_str(web_search_context)
            messages.append(("system", f"Here are the web search results for the relevant keywords. Treat these results as background knowledge only:\n{web_search_context_str}"))

        return super().invoke(messages)


relevency_filter_agent = RelevencyFilterAgent(
    deploy_method=settings.DEPLOY_METHOD,
    base_url=settings.BASE_URL
)

