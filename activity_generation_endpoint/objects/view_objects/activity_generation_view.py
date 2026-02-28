from activity_generation_endpoint.objects.activity_generation.activity_generation_agent import (
    activity_generation_agent,
)
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status


class ActivityGenerationView(APIView):
    """
    Handle activity generation requests.

    The view validates required case/task fields, optionally enriches the prompt
    with web-search context, and returns generated activity content.

    Required request fields:
    - case_title
    - case_description
    - task_title
    - task_description

    Optional request fields:
    - web_search: bool | str

    Responses:
    - 200: {"result": <generated activity>, "sources": [<source URLs>]}
    - 400: Validation/formatting error details
    """

    # Names of required fields in the incoming request data
    REQUIRED_FIELDS = (
        "case_title",
        "case_description",
        "task_title",
        "task_description",
    )

    def post(self, request, *args, **kwargs):
        """
        Process a POST request to generate an activity.

        Parameters
        ----------
        request : rest_framework.request.Request
            The HTTP request containing case and task data.
        *args, **kwargs
            Additional positional and keyword arguments passed by the
            DRF router.

        Returns
        -------
        rest_framework.response.Response
            200 OK with {"result": activity_data, "sources": [source URLs]}
            on success, or 400 Bad Request with error details.
        """
        data = request.data

        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in data]
        empty_fields = [
            f
            for f in self.REQUIRED_FIELDS
            if f in data and str(data.get(f)).strip() == ""
        ]

        if missing_fields or empty_fields:
            return Response(
                {
                    "error": "Invalid parameters",
                    "missing_fields": missing_fields,
                    "empty_fields": empty_fields,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        case_title = data.get("case_title")
        case_description = data.get("case_description")
        task_title = data.get("task_title")
        task_description = data.get("task_description")
        web_search_enabled = data.get("web_search", False)

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

        context = None
        sources = set()

        if web_search_enabled:
            query = "\n\n".join(
                [
                    f"Case Title: {case_title}",
                    f"Case Description: {case_description}",
                    f"Task Title: {task_title}",
                    f"Task Description: {task_description}",
                ]
            )
            searcher = WebSearcher()
            context = searcher.run(query)
            for entry in context.values():
                sources.update(entry["sources"])

        activity_data = activity_generation_agent.invoke(
            case_title=case_title,
            case_description=case_description,
            task_title=task_title,
            task_description=task_description,
            web_search_context=context,
        )

        return Response(
            {"result": activity_data, "sources": list(sources)},
            status=status.HTTP_200_OK,
        )