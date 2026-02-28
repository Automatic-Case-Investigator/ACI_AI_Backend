import logging
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from relevency_filter_endpoint.objects.relevency_filter.relevency_filter_agent import relevency_filter_agent
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher

logger = logging.getLogger(__name__)


class RelevencyFilterView(APIView):
    """
    API endpoint for filtering SIEM query relevancy.
    """

    # Required fields that must be present in the POST payload.
    REQUIRED_FIELDS = [
        "case_title",
        "case_description",
        "task_title",
        "task_description",
        "activity",
        "query",
        "event",
    ]

    def post(self, request, *args, **kwargs):
        """
        Handle POST request to evaluate SIEM query relevancy.

        Parameters
        ----------
        request : rest_framework.request.Request
            Incoming request containing SIEM data.
        *args, **kwargs
            Additional arguments passed by the router.

        Returns
        -------
        rest_framework.response.Response
            200 OK with relevancy information and sources on success,
            400 Bad Request if required fields are missing or if
            the web_search parameter is malformed.
        """
        data = request.data

        # Validates missing fields
        missing_fields = [field for field in self.REQUIRED_FIELDS if not data.get(field)]

        if missing_fields:
            return Response(
                {
                    "error": "Required fields missing",
                    "missing_fields": missing_fields,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        siem = data.get("siem")
        case_title = data.get("case_title")
        case_description = data.get("case_description")
        task_title = data.get("task_title")
        task_description = data.get("task_description")
        activity = data.get("activity")
        query = data.get("query")
        event = data.get("event")
        web_search_enabled = data.get("web_search", False)

        # Validates web_search flag
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

        # Run web searher
        context = {}
        sources = set()

        if web_search_enabled:
            searcher = WebSearcher()

            search_prompt = (
                f"Case Title: {case_title}\n\n"
                f"Case Description: {case_description}\n\n"
                f"Task Title: {task_title}\n\n"
                f"Task Description: {task_description}\n"
                f"Activity: {activity}\n"
                f"SIEM Query: {query}\n"
                f"Event Queried: {event}"
            )

            context = searcher.run(search_prompt)

            # Safely aggregate sources
            if isinstance(context, dict):
                for keyword_data in context.values():
                    if isinstance(keyword_data, dict):
                        sources.update(keyword_data.get("sources", []))

        # Invoke relevancy agent
        relevency_info = relevency_filter_agent.invoke(
            siem=siem,
            query=query,
            event=event,
            case_title=case_title,
            case_description=case_description,
            task_title=task_title,
            task_description=task_description,
            activity=activity,
            web_search_context=context,
        )

        # Success response
        return Response(
            {
                "result": relevency_info,
                "sources": list(sources),
            },
            status=status.HTTP_200_OK,
        )
