from task_generation_endpoint.objects.task_generation.task_generation_agent import (
    task_generation_agent,
)
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status


class TaskGenerationView(APIView):
    """
    Handles task generation requests via POST.

    The view extracts case information, optionally performs a web search,
    delegates generation to `task_generation_agent.invoke`, and returns
    the generated task along with any source identifiers.

    Attributes
    ----------
    REQUIRED_FIELDS : list
        The names of required fields in the incoming request data.
    """

    # Required fields for POST request
    REQUIRED_FIELDS = ["case_title", "case_description"]

    def post(self, request, *args, **kwargs):
        """
        Process a POST request to generate a task.

        Parameters
        ----------
        request : rest_framework.request.Request
            The HTTP request containing case data.
        *args, **kwargs
            Additional positional and keyword arguments passed by the
            DRF router.

        Returns
        -------
        rest_framework.response.Response
            200 OK with {"result": task_data, "sources": [source_id, ...]}
            on success, or 400 Bad Request with error details.
        """
        data = request.data

        missing_fields = []
        empty_fields = []

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
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        case_title = data.get("case_title")
        case_description = data.get("case_description")
        web_search_enabled = data.get("web_search", False)

        # Validate and normalise the web_search flag
        if isinstance(web_search_enabled, str):
            if web_search_enabled.isdigit():
                web_search_enabled = bool(int(web_search_enabled))
            elif web_search_enabled.lower() in ["true", "false"]:
                web_search_enabled = web_search_enabled.lower() == "true"
            else:
                return Response(
                    {"error": 'Parameter "web_search" is incorrectly formatted'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        context = None
        sources = set()

        if web_search_enabled:
            searcher = WebSearcher()
            context = searcher.run(
                f"Case Title: {case_title}\n\nCase Description: {case_description}\n"
            )

            for keyword in context.keys():
                sources.update(context[keyword]["sources"])

        task_data = task_generation_agent.invoke(
            case_title=case_title,
            case_description=case_description,
            web_search_context=context,
        )

        return Response(
            {"result": task_data, "sources": list(sources)},
            status=status.HTTP_200_OK,
        )
