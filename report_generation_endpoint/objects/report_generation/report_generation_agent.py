import json

from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM


with open(settings.REPORT_GENERATION_CONFIG_PATH, "r") as file:
    _CONFIG = json.load(file)


class ReportGenerationAgent(LLM):
    """Agent for generating activity, task, and case reports."""

    def __init__(self, deploy_method: str, base_url: str):
        super().__init__(
            deploy_method=deploy_method,
            model_name=_CONFIG["model_name"],
            base_url=base_url,
            reasoning_effort="low",
            temperature=0.4,
        )

    def invoke(self, case_title: str, case_description: str, task_title: str, task_description: str, activity: str, report_template: str) -> str:
        messages = [
            ("system", _CONFIG["instruction"]["activity"]),
            ("human", f"Case title:\n{case_title}\n---"),
            ("human", f"Case description:\n{case_description}\n---"),
            ("human", f"Task title:\n{task_title}\n---"),
            ("human", f"Task description:\n{task_description}\n---"),
            ("human", f"Activity:\n{activity}\n---"),
            ("human", f"Report template:\n```\n{report_template}\n```"),
        ]
        return super().invoke(messages)

    def invoke_task(self, case_title: str, case_description: str, task_title: str, task_description: str, activity_reports: list[str]) -> str:
        activity_reports_text = "---\n\n" + "\n\n---\n\n".join(activity_reports) + "\n\n---"
        messages = [
            ("system", _CONFIG["instruction"]["task"]),
            ("human", f"Case title:\n{case_title}\n---"),
            ("human", f"Case description:\n{case_description}\n---"),
            ("human", f"Task title:\n{task_title}\n---"),
            ("human", f"Task description:\n{task_description}\n---"),
            ("human", f"Activity reports:\n{activity_reports_text}\n---"),
        ]
        return super().invoke(messages)

    def invoke_case(self, case_title: str, case_description: str, task_reports: list[str], report_template: str) -> str:
        task_reports_text = "---\n\n" + "\n\n---\n\n".join(task_reports) + "\n\n---"
        print(task_reports_text)
        messages = [
            ("system", _CONFIG["instruction"]["case"]),
            ("human", f"Case title:\n{case_title}\n---"),
            ("human", f"Case description:\n{case_description}\n---"),
            ("human", f"Task reports:\n{task_reports_text}\n---"),
            ("human", f"Report template:\n```\n{report_template}\n```"),
        ]
        return super().invoke(messages)

report_generation_agent = ReportGenerationAgent(
    deploy_method=settings.DEPLOY_METHOD,
    base_url=settings.BASE_URL,
)
