from ACI_AI_Backend import settings
from ACI_AI_Backend.llm import LLM
import json

from ACI_AI_Backend.objects.prompt_injection_detection.prompt_injection_detector import PromptInjectionDetector
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher

_CONFIG = json.load(open(settings.QUERY_GENERATION_CONFIG_PATH, "r"))


class QueryGenerationAgent(LLM):
    """
    Generates queries for a SIEM system using the configured LLM.

    Attributes
    ----------
    rejection_message : str
        Message returned when the user prompt is detected as unsafe.
    """

    __slots__ = ("rejection_message",)

    def __init__(self, deploy_method: str, base_url: str):
        """
        Initialise the QueryGenerationAgent.

        Parameters
        ----------
        deploy_method : str
            Deployment method for the LLM.
        base_url : str
            Base URL for the LLM service.

        Returns
        -------
        None
        """
        self.rejection_message = "Could not provide answer to the given instructions"
        super().__init__(deploy_method=deploy_method, model_name=_CONFIG["model_name"], base_url=base_url, reasoning_effort="medium", temperature=0.4)

    def fix_queries(self, siem: str = "wazuh", input_str: str = "The user did not provide any inputs"):
        """
        Returns the syntacticly corrected of generated query. Returns "NOT QUERY"
        if input_str does not contain queries

        Parameters
        ----------
        siem : str
            The name of the SIEM system (currently only 'wazuh' is supported)
        input_str : str
            The text input to correct query syntax for

        Returns
        -------
        str
            The LLM response of the corrected syntax

        """
        messages = []

        if siem == "splunk":
            pass
        else:
            print(f"Fixing queries: {input_str}")
            prompt = _CONFIG["instruction"]["open_search"]["fix_query"]
            messages = [
                ("system", prompt),
                ("human", f"Input: {input_str}"),
            ]

        return super().invoke(messages)

    def invoke(
        self,
        siem: str = "wazuh",
        user_prompt: str | None = None,
        case_title: str | None = None,
        case_description: str | None = None,
        task_title: str | None = None,
        task_description: str | None = None,
        activity: str | None = None,
        fields: str | None = None,
        prev_activity_critique: str | None = None,
        max_queries_per_iteration: int | None = None,
        earliest_unit: str | None = None,
        earliest_magnitude: int | None = None,
        vicinity_unit: str | None = None,
        vicinity_magnitude: int | None = None,
        additional_notes: str | None = None,
        relevant_config_file_contents: dict | None = None,
        web_search_context: dict | None = None,
    ):
        """
        Generate a query based on the provided context or a user prompt.

        Parameters
        ----------
        siem : str, default 'wazuh'
            The name of the SIEM system (currently only 'wazuh' is supported).
        user_prompt : str | None
            A free‑form prompt supplied by the user.
        case_title : str | None
            Title of the case.
        case_description : str | None
            Description of the case.
        task_title : str | None
            Title of the task.
        task_description : str | None
            Description of the task.
        activity : str | None
            Activity context for the query.
        fields : str | None
            String containing all the fields and their respective type in the SIEM
        prev_activity_critique : str | None
            Critique of the previous activity investigation
        max_queries_per_iteration : int | None
            Maximum number of queries the model may generate for this iteration.
        earliest_unit : str | None
            Unit of time for the earliest time from now to find events
        earliest_magnitude : int | None
            Magnitude of time for the earliest time from now to find events
        vicinity_unit : str | None
            Unit of time for the time window of "close together" events
        vicinity_magnitude: int | None
            Magnitude of time for the time window of "close together" events
        additional_notes : str | None
            Additional notes from the human SOC analyst expert for the investigation.
        relevant_config_file_contents : dict | None
            Mapping of config filename to relevant SIEM config file contents.
        web_search_context : dict | None
            Optional web search results to provide additional context.

        Returns
        -------
        str
            The LLM response or a rejection message if the prompt is unsafe.
        """
        messages = []

        if siem == "splunk":
            pass
        else:
            if user_prompt:
                injection_detector = PromptInjectionDetector()
                if not injection_detector.is_safe(user_prompt):
                    return self.rejection_message
                prompt = _CONFIG["instruction"]["open_search"]["from_prompt"]
                messages = [("system", prompt), ("human", user_prompt)]
            elif all([case_title, case_description, task_title, task_description, activity]):
                print("Generating query from case\n")

                prompt = _CONFIG["instruction"]["open_search"]["from_case"]
                
                if prev_activity_critique:
                    # Consider a specialized prompt for regenerating queries
                    prompt = _CONFIG["instruction"]["open_search"]["from_case_regenerate"]
 

                messages = [
                    ("system", prompt),
                ]

                if max_queries_per_iteration is not None:
                    messages.append(
                        (
                            "system",
                            f"Constraint: Generate at most {max_queries_per_iteration} queries in this iteration. Never exceed this limit.",
                        )
                    )

                if earliest_unit is not None and earliest_magnitude is not None:
                    messages.append(
                        (
                            "system",
                            f"Constraint: When you are asked to identify recent events, Only include events from the past {earliest_magnitude} {earliest_unit}. Exclude anything outside this range.",
                        )
                    )

                if vicinity_unit is not None and vicinity_magnitude is not None:
                    messages.append(
                        (
                            "system",
                            f"Constraint: When you are not asked to identify recent events, but instead asked to find events near a timestamp, use a ±{vicinity_magnitude} {vicinity_unit} window. Do not use a different range.",
                        )
                    )

                messages.extend(
                    [
                        ("human", f"# Case title:\n{case_title}\n---"),
                        ("human", f"# Case description:\n{case_description}\n---"),
                        ("human", f"# Task title:\n{task_title}\n---"),
                        ("human", f"# Task description:\n{task_description}\n---"),
                        ("human", f"# Activity:\n{activity}\n---"),
                    ]
                )

                if fields:
                    print(fields)
                    messages.append(("human", f"# Available fields in SIEM:\nThe following text contain all the available fields in the SIEM, followed by their corresponding information:\n{fields}\n---"))

                if prev_activity_critique:
                    messages.append(("human", f"# Critique from previous investigation:\n{prev_activity_critique}\n---"))
            else:
                raise ValueError("Invalid arguments for invoke()")

        if additional_notes is not None:
            messages.append(("human", f"# Additional notes from the human SOC analyst expert for the investigation.\n{additional_notes}\n---"))

        if relevant_config_file_contents:
            contents_str = ""
            for filename, content in relevant_config_file_contents.items():
                contents_str += f"{filename}: ```\n{content}\n```\n\n"

            print(contents_str)
            messages.append(("human", f"# Relevant SIEM config file contents:\n{contents_str}\n---"))

        if web_search_context:
            web_search_context_str = WebSearcher.context_to_str(web_search_context)
            messages.append(("system", f"Here are the web search results for the relevant keywords. Treat these results as background knowledge only:\n{web_search_context_str}"))

        return super().invoke(messages)


query_generation_agent = QueryGenerationAgent(deploy_method=settings.DEPLOY_METHOD, base_url=settings.BASE_URL)
