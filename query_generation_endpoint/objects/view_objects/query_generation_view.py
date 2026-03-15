from query_generation_endpoint.objects.query_generation.query_generation_agent import (
    query_generation_agent,
)
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status


class QueryGenerationView(APIView):
    """
    Handle SIEM query generation requests.

    Supports two request modes:
    - **Prompt mode** – supply ``prompt`` (optional ``siem`` and ``web_search``).
    - **Structured mode** – supply the required case/task/activity fields when ``prompt`` is omitted.

    Required structured fields:
        case_title, case_description,
        task_title, task_description, activity

    Optional fields:
        prompt (str), siem (str), web_search (bool | str)

    Responses:
        200: {"result": <generated query>, "sources": [<source URLs>]}
        400: Validation/formatting or unsupported SIEM error details
    """

    # Fields that must be present in structured mode
    REQUIRED_FIELDS = [
        "case_title",
        "case_description",
        "task_title",
        "task_description",
        "activity",
        "fields",
    ]

    def post(self, request, *args, **kwargs):
        data = request.data

        user_prompt = data.get("prompt")
        siem = data.get("siem")

        # -------------------------------------------------
        # Validation of structured fields (when no prompt)
        # -------------------------------------------------
        missing_fields: list[str] = []
        empty_fields: list[str] = []

        if not user_prompt:
            for field in self.REQUIRED_FIELDS:
                if field not in data:
                    missing_fields.append(field)
                elif str(data.get(field)).strip() == "":
                    empty_fields.append(field)

        if missing_fields or empty_fields:
            return Response(
                {
                    "error": "Invalid parameters",
                    "missing_fields": missing_fields,
                    "empty_fields": empty_fields,
                    "note": "Structured fields are required when 'prompt' is not provided",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # -------------------------------------------------
        # Extract common fields
        # -------------------------------------------------
        case_title = data.get("case_title")
        case_description = data.get("case_description")
        task_title = data.get("task_title")
        task_description = data.get("task_description")
        activity = data.get("activity")
        fields = data.get("fields")
        prev_activity_critique = data.get("prev_activity_critique", None)
        web_search_enabled = data.get("web_search", False)

        # -------------------------------------------------
        # Normalise web_search flag
        # -------------------------------------------------
        if isinstance(web_search_enabled, str):
            lw = web_search_enabled.lower()
            if lw.isdigit():
                web_search_enabled = bool(int(lw))
            elif lw in {"true", "false"}:
                web_search_enabled = lw == "true"
            else:
                return Response(
                    {"error": 'Parameter "web_search" is incorrectly formatted'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if prev_activity_critique is not None:
            if not isinstance(prev_activity_critique, str):
                return Response(
                    {"error": 'Parameter "prev_activity_critique" must be a string'},
                    status=status.HTTP_400_BAD_REQUEST,
                )


        # -------------------------------------------------
        # Build optional web‑search context
        # -------------------------------------------------
        context: dict | None = None
        sources: set[str] = set()

        if user_prompt and web_search_enabled:
            searcher = WebSearcher()
            context = searcher.run(user_prompt)
            for keyword in context.keys():
                sources.update(context[keyword]["sources"])

        if not user_prompt and web_search_enabled:
            # Structured query – build a composite prompt for web search
            search_query = (
                f"Case Title: {case_title}\n\n"
                f"Case Description: {case_description}\n\n"
                f"Task Title: {task_title}\n\n"
                f"Task Description: {task_description}\n\n"
                f"Activity: {activity}"
            )
            searcher = WebSearcher()
            context = searcher.run(search_query)
            for keyword in context.keys():
                sources.update(context[keyword]["sources"])

        # -------------------------------------------------
        # Invoke the query generation agent
        # -------------------------------------------------
        try:
            if user_prompt:
                query_data = query_generation_agent.generate(
                    siem=siem,
                    user_prompt=user_prompt,
                    web_search_context=context,
                )
            else:
                query_data = query_generation_agent.generate(
                    siem=siem,
                    case_title=case_title,
                    case_description=case_description,
                    task_title=task_title,
                    task_description=task_description,
                    activity=activity,
                    fields=fields,
                    prev_activity_critique=prev_activity_critique,
                    web_search_context=context,
                )
        except ValueError as e:
            # Capture unsupported SIEM or argument errors
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # -------------------------------------------------
        # Final response
        # -------------------------------------------------
        if not query_data:
            # The agent returned an empty string / None → unsupported SIEM
            return Response(
                {"error": "Specified SIEM platform not implemented"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {"result": query_data, "sources": list(sources)},
            status=status.HTTP_200_OK,
        )
