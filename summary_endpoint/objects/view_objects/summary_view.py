from typing import Any, Dict, List
from collections import Counter

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from summary_endpoint.objects.summary.summary_agent import summary_agent


class QuerySummaryView(APIView):
    """
    Handle query summarization requests.

    Reads a SIEM query and events queried by it, then returns a summary of the query's intent and results.
    
    Required request fields:
    - query
    - events

    Responses:
    - 200: {"result": <query summary>}
    - 400: Validation/formatting error details
    """
    def post(self, request, *args, **kwargs):
        payload = request.data or {}
        query = payload.get("query")
        events = payload.get("events")

        print(events)

        if not isinstance(query, str) or not query.strip():
            return Response(
                {"error": "Missing or invalid 'query' (string)."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not isinstance(events, str):
            return Response(
                {"error": "Missing or invalid 'events' (string)."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        summary = summary_agent.summarize_query(query=query, events=events)
        return Response({"result": summary}, status=status.HTTP_200_OK)


def _split_multiline_or_comma(s: str) -> List[str]:
    """Split a string by newlines and commas into cleaned non-empty entries."""
    if not s:
        return []
    parts: List[str] = []
    for line in s.splitlines():
        for seg in line.split(","):
            item = seg.strip()
            if item:
                parts.append(item)
    return parts


class ActivitySummaryView(APIView):
    def post(self, request, *args, **kwargs):
        payload = request.data or {}
        required = [
            "case_title",
            "case_description",
            "task_title",
            "task_description",
            "activity",
            "queries",
            "query_summaries",
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
        queries_raw = payload.get("queries")
        summaries_raw = payload.get("query_summaries")

        queries_list = _split_multiline_or_comma(queries_raw)
        summaries_list = _split_multiline_or_comma(summaries_raw)

        activity_summary = summary_agent.summarize_activity(
            case_title=case_title,
            case_description=case_description,
            task_title=task_title,
            task_description=task_description,
            activity=activity,
            queries=queries_list,
            query_summaries=summaries_list,
        )

        return Response({"result": activity_summary}, status=status.HTTP_200_OK)
