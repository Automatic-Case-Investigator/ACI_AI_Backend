import json
from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher

with open(settings.COMPLETION_CHECK_CONFIG_PATH, "r") as file:
    _CONFIG = json.load(file)


class CompletionCheckAgent(LLM):
    def __init__(self, deploy_method: str, base_url: str):
        self.prompt = _CONFIG["instruction"]
        super().__init__(deploy_method=deploy_method, model_name=_CONFIG["model_name"], base_url=base_url)

    def invoke(
            self,
            case_title: str,
            case_description: str,
            task_title: str,
            task_description: str,
            activity: str,
            queries: list[str],
            query_summaries: list[str],
            available_siem_fields: str | None = None,
            additional_notes: str | None = None,
            web_search_context: dict | None = None,
    ) -> str:
        messages = [
            ("system", self.prompt),
            ("human", f"# Case title:\n{case_title}\n---"),
            ("human", f"# Case description:\n{case_description}\n---"),
            ("human", f"# Task title:\n{task_title}\n---"),
            ("human", f"# Task description:\n{task_description}\n---"),
            ("human", f"# Activity:\n{activity}\n---"),
        ]

        for query, summary in zip(queries, query_summaries):
            messages.append(("human", f"# Query:\n{query}\n---\n# Summary:\n{summary}\n---"))
        
        if additional_notes is not None:
            messages.append(("human", f"# Additional notes from the human SOC analyst expert for the investigation.\n{additional_notes}\n---"))

        if available_siem_fields is not None:
            messages.append(("human", f"# Available SIEM fields and their information:\n{available_siem_fields}\n---"))

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


completion_check_agent = CompletionCheckAgent(deploy_method=settings.DEPLOY_METHOD, base_url=settings.BASE_URL)
