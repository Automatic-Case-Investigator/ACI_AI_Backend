import json
from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher

with open(settings.SUMMARY_CONFIG_PATH, "r") as file:
    _CONFIG = json.load(file)


class SummaryAgent(LLM):
    """SummaryAgent generates summaries for queries and activities.

    Parameters:
        deploy_method (str): Deployment method for the LLM.
        base_url (str): Base URL for the LLM service.
    """

    def __init__(self, deploy_method: str, base_url: str):
        """Initialize the agent with configuration.

        Parameters:
            deploy_method (str): Deployment method for the LLM.
            base_url (str): Base URL for the LLM service.

        Returns:
            None
        """
        self.prompt = _CONFIG["instruction"]
        super().__init__(deploy_method=deploy_method, model_name=_CONFIG["model_name"], base_url=base_url, reasoning_effort="low")

    def summarize_query(
        self,
        query: str,
        events: str,
        additional_notes: str | None = None,
        web_search_context: dict | None = None,
    ) -> str:
        """Summarize a query and its associated events.

        Parameters:
            query (str): The query to summarize.
            events (str): The events related to the query.
            additional_notes (str | None): Additional notes from the human SOC analyst expert for the investigation.
            web_search_context (dict or None): Optional web search context.

        Returns:
            str: The LLM's output summarizing the query and events.
        """
        messages = [
            ("system", self.prompt["query_summary"]),
            ("user", f"# Query:\n{query}\n---"),
            ("user", f"# Events:\n{events}\n---")
        ]

        if additional_notes is not None:
            messages.append(("user", f"# Additional notes from the human SOC analyst expert for the investigation.\n{additional_notes}\n---"))

        if web_search_context:
            web_search_context_str = WebSearcher.context_to_str(web_search_context)
            messages.append(("system", f"Here are the web search results for the relevant keywords. Treat these results as background knowledge only:\n{web_search_context_str}"))

        # Use the helper invoke method to get the LLM's output
        return super().invoke(messages)

    def summarize_activity(
        self,
        case_title: str,
        case_description: str,
        task_title: str,
        task_description: str,
        activity: str,
        queries: list[str],
        query_summaries: list[str],
        additional_notes: str | None = None,
        web_search_context: dict | None = None,
    ) -> str:
        """Summarize activity details including queries and their summaries.

        Parameters:
            case_title (str): Title of the case.
            case_description (str): Description of the case.
            task_title (str): Title of the task.
            task_description (str): Description of the task.
            activity (str): Activity description.
            queries (list[str]): List of queries.
            query_summaries (list[str]): Corresponding summaries of the queries.
            additional_notes (str | None): Additional notes from the human SOC analyst expert for the investigation.
            web_search_context (dict or None): Optional web search context.

        Returns:
            str: The LLM's output summarizing the activity.
        """
        messages = [
            ("system", self.prompt["activity_summary"]),
            ("human", f"# Case title:\n{case_title}\n---"),
            ("human", f"# Case description:\n{case_description}\n---"),
            ("human", f"# Task title:\n{task_title}\n---"),
            ("human", f"# Task description:\n{task_description}\n---"),
            ("human", f"# Activity:\n{activity}\n---"),
        ]

        for query, summary in zip(queries, query_summaries):
            messages.append(("human", f"# Query:\n{query}\n# Summary:\n{summary}\n---"))
        if additional_notes is not None:
            messages.append(("human", f"# Additional notes from the human SOC analyst expert for the investigation.\n{additional_notes}\n---"))

        if web_search_context:
            parts = []
            for keyword, data in web_search_context.items():
                explanation = data.get("explanation", "")
                if explanation:
                    parts.append(f"{keyword}:\n```\n{explanation}\n```\n\n")
            if parts:
                web_search_context_str = "".join(parts)
                messages.append(("system", f"Here are the web search results for the relevant keywords. Treat these results as background knowledge only:\n{web_search_context_str}"))
        return super().invoke(messages)


summary_agent = SummaryAgent(deploy_method=settings.DEPLOY_METHOD, base_url=settings.BASE_URL)
