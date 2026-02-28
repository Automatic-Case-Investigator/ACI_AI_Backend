from typing import Any, Dict, List
from collections import Counter

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from completion_check_endpoint.objects.completion_check.completion_check_agent import completion_check_agent


class ActivityCompletionCheckView(APIView):
    """
    Handle activity completion check requests.

    Validates desired case/task/activity details along with associated queries and summaries, then returns an assessment of activity completeness.

    Required request fields:
    - case_title
    - case_description
    - task_title
    - task_description
    - activity
    - queries
    - query_summaries

    Responses:
    - 200: {"result": <completeness assessment>}
    - 400: Validation/formatting error details
    """

    def post(self, request, *args, **kwargs):
        payload = request.data or {}
        required = [
            "case_title",
            "case_description",
            "task_title",
            "task_description",
            "activity",
            "queries",
            "query_summaries"
        ]


        missing = []
        empty = []
        for f in required:
            if f not in payload:
                missing.append(f)
            elif not isinstance(payload.get(f), str) or not payload.get(f).strip():
                empty.append(f)

        if missing or empty:
            return Response(
                {
                    "error": "Invalid parameters",
                    "missing_fields": missing,
                    "empty_fields": empty,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        case_title = payload.get("case_title").strip()
        case_description = payload.get("case_description").strip()
        task_title = payload.get("task_title").strip()
        task_description = payload.get("task_description").strip()
        activity = payload.get("activity").strip()
        queries = payload.getlist("queries")
        query_summaries = payload.getlist("query_summaries")

        response = completion_check_agent.check_activity_completeness(
            case_title=case_title,
            case_description=case_description,
            task_title=task_title,
            task_description=task_description,
            activity=activity,
            queries=queries,
            query_summaries=query_summaries
        )

        return Response({"result": response}, status=status.HTTP_200_OK)
