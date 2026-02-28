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

    def check_activity_completeness(
            self, case_title: str, case_description: str, task_title: str, task_description: str, activity: str, queries: list[str], query_summaries: list[str], web_search_context: dict | None = None
    ) -> str:
        messages = [
            ("system", self.prompt),
            ("human", f"Case title: {case_title}"),
            ("human", f"Case description: {case_description}"),
            ("human", f"Task title: {task_title}"),
            ("human", f"Task description: {task_description}"),
        ]

        for query, summary in zip(queries, query_summaries):
            messages.append(("human", f"Query: {query}\nSummary: {summary}"))

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
