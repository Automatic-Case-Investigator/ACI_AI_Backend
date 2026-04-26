from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from report_generation_endpoint.objects.report_generation.report_generation_agent import (
    report_generation_agent,
)


def _is_empty_value(value) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    return False


def _validate_string_field(data, field: str, missing_fields: list[str], empty_fields: list[str], invalid_type_fields: list[str]) -> None:
    if field not in data:
        missing_fields.append(field)
        return

    value = data.get(field)
    if not isinstance(value, str):
        invalid_type_fields.append(field)
        return

    if _is_empty_value(value):
        empty_fields.append(field)


def _validate_string_array_field(data, field: str, missing_fields: list[str], empty_fields: list[str], invalid_type_fields: list[str]) -> None:
    if field not in data:
        missing_fields.append(field)
        return

    value = data.get(field)
    if not isinstance(value, list):
        invalid_type_fields.append(field)
        return

    if not all(isinstance(item, str) and item.strip() != "" for item in value):
        invalid_type_fields.append(field)


class ActivityReportGenerator(APIView):
    REQUIRED_FIELDS = [
        "case_title",
        "case_description",
        "task_title",
        "task_description",
        "activity",
        "report_template",
    ]

    def post(self, request, *args, **kwargs):
        data = request.data

        missing_fields: list[str] = []
        empty_fields: list[str] = []
        invalid_type_fields: list[str] = []

        for field in self.REQUIRED_FIELDS:
            _validate_string_field(data, field, missing_fields, empty_fields, invalid_type_fields)

        if missing_fields or empty_fields or invalid_type_fields:
            return Response(
                {
                    "error": "Invalid parameters",
                    "missing_fields": missing_fields,
                    "empty_fields": empty_fields,
                    "invalid_type_fields": invalid_type_fields,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = report_generation_agent.invoke(
            case_title=data.get("case_title"),
            case_description=data.get("case_description"),
            task_title=data.get("task_title"),
            task_description=data.get("task_description"),
            activity=data.get("activity"),
            report_template=data.get("report_template"),
        )

        return Response({"result": result}, status=status.HTTP_200_OK)


class TaskReportGenerator(APIView):
    REQUIRED_FIELDS = [
        "case_title",
        "case_description",
        "task_title",
        "task_description",
        "activity_reports",
    ]

    def post(self, request, *args, **kwargs):
        data = request.data
        activity_reports = data.getlist("activity_reports") if hasattr(data, "getlist") else data.get("activity_reports")

        missing_fields: list[str] = []
        empty_fields: list[str] = []
        invalid_type_fields: list[str] = []

        for field in ["case_title", "case_description", "task_title", "task_description"]:
            _validate_string_field(data, field, missing_fields, empty_fields, invalid_type_fields)

        if hasattr(data, "getlist") and "activity_reports" in data:
            if len(activity_reports) == 0:
                empty_fields.append("activity_reports")
            elif not all(isinstance(item, str) and item.strip() != "" for item in activity_reports):
                invalid_type_fields.append("activity_reports")
        else:
            _validate_string_array_field(data, "activity_reports", missing_fields, empty_fields, invalid_type_fields)

        if missing_fields or empty_fields or invalid_type_fields:
            print({
                    "error": "Invalid parameters",
                    "missing_fields": missing_fields,
                    "empty_fields": empty_fields,
                    "invalid_type_fields": invalid_type_fields,
                })
            return Response(
                {
                    "error": "Invalid parameters",
                    "missing_fields": missing_fields,
                    "empty_fields": empty_fields,
                    "invalid_type_fields": invalid_type_fields,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = report_generation_agent.invoke_task(
            case_title=data.get("case_title"),
            case_description=data.get("case_description"),
            task_title=data.get("task_title"),
            task_description=data.get("task_description"),
            activity_reports=activity_reports,
        )

        return Response({"result": result}, status=status.HTTP_200_OK)


class CaseReportGenerator(APIView):
    REQUIRED_FIELDS = [
        "case_title",
        "case_description",
        "task_reports",
        "report_template",
    ]

    def post(self, request, *args, **kwargs):
        data = request.data
        task_reports = data.getlist("task_reports") if hasattr(data, "getlist") else data.get("task_reports")

        missing_fields: list[str] = []
        empty_fields: list[str] = []
        invalid_type_fields: list[str] = []

        for field in ["case_title", "case_description", "report_template"]:
            _validate_string_field(data, field, missing_fields, empty_fields, invalid_type_fields)

        if hasattr(data, "getlist") and "task_reports" in data:
            if len(task_reports) == 0:
                empty_fields.append("task_reports")
            elif not all(isinstance(item, str) and item.strip() != "" for item in task_reports):
                invalid_type_fields.append("task_reports")
        else:
            _validate_string_array_field(data, "task_reports", missing_fields, empty_fields, invalid_type_fields)

        if missing_fields or empty_fields or invalid_type_fields:
            print({
                "error": "Invalid parameters",
                "missing_fields": missing_fields,
                "empty_fields": empty_fields,
                "invalid_type_fields": invalid_type_fields,
            })  

            return Response(
                {
                    "error": "Invalid parameters",
                    "missing_fields": missing_fields,
                    "empty_fields": empty_fields,
                    "invalid_type_fields": invalid_type_fields,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        result = report_generation_agent.invoke_case(
            case_title=data.get("case_title"),
            case_description=data.get("case_description"),
            task_reports=task_reports,
            report_template=data.get("report_template"),
        )

        return Response({"result": result}, status=status.HTTP_200_OK)
