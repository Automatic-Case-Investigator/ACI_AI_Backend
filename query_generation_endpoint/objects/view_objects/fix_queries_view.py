from query_generation_endpoint.objects.query_generation.query_generation_agent import (
    query_generation_agent,
)
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status


class FixQueriesView(APIView):
    """
    Handle SIEM query syntax fixes.

    Required fields:
        input_str (str)

    Optional fields:
        siem (str)

    Responses:
        200: {"result": <generated query>}
        400: Validation/formatting or unsupported SIEM error details
    """

    # Fields that must be present in structured mode
    REQUIRED_FIELDS = [
        "input_str",
    ]

    def post(self, request, *args, **kwargs):
        data = request.data

        # -------------------------------------------------
        # Validation of structured fields 
        # -------------------------------------------------
        missing_fields: list[str] = []
        empty_fields: list[str] = []

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
        # Extract fields
        # -------------------------------------------------
        input_str = data.get("input_str")
        siem = data.get("siem")

        if input_str is not None and not isinstance(input_str, str):
            return Response(
                {"error": 'Parameter "input_str" is incorrectly formatted'},
                status=status.HTTP_400_BAD_REQUEST,
             )

        # -------------------------------------------------
        # Invoke the query generation agent
        # -------------------------------------------------
        try:
            corrected_response = query_generation_agent.fix_queries(
                siem=siem,
                input_str=input_str
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
        if not corrected_response:
            # The agent returned an empty string / None → unsupported SIEM
            return Response(
                {"error": "Specified SIEM platform not implemented"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {"result": corrected_response},
            status=status.HTTP_200_OK,
        )
