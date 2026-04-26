from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
import json


_CONFIG = json.load(open(settings.CONFIG_RETRIEVAL_QUERY_CONFIG_PATH, "r"))


class ConfigRetrievalQueryAgent(LLM):
    """Generate a focused SIEM-config retrieval query from an investigation activity."""

    def __init__(self, deploy_method: str, base_url: str):
        super().__init__(
            deploy_method=deploy_method,
            model_name=_CONFIG["model_name"],
            base_url=base_url,
            reasoning_effort="low",
        )

    def invoke(self, investigation_activity: str) -> str:
        """Return a single focused query for SIEM config-file vector retrieval."""
        messages = [
            ("system", _CONFIG["instruction"]),
            ("human", f"Investigation activity:\n{investigation_activity}"),
        ]

        return super().invoke(messages).strip()


config_retrieval_query_agent = ConfigRetrievalQueryAgent(
    deploy_method=settings.DEPLOY_METHOD,
    base_url=settings.BASE_URL,
)
