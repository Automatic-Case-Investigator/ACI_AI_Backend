from query_generation_endpoint.objects.query_generation.query_generation_trainer import (
    QueryGenerationTrainer,
)
from query_generation_endpoint.objects.query_generation.query_generator import (
    query_generator,
)
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from ACI_AI_Backend.objects.redis_client import redis_client
from rest_framework.response import Response
from query_generation_endpoint.models import *
from rest_framework.views import APIView
from rest_framework import status
from django.conf import settings
import json

file = open(settings.QUERY_GENERATION_CONFIG_PATH, "r")
config = json.load(file)
file.close()


class QueryGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        user_prompt = request.POST.get("prompt")
        case_title = request.POST.get("case_title")
        case_description = request.POST.get("case_description")
        task_title = request.POST.get("task_title")
        task_description = request.POST.get("task_description")
        activity = request.POST.get("activity")
        siem = request.POST.get("siem")

        if (task_title is None or task_description is None) and user_prompt is None:
            return Response(
                {"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST
            )
            
        query_data = None
        if user_prompt:
            # Defaults to generate from prompt if it is provided
            query_data = query_generator.generate_query_from_prompt(
                is_splunk=siem == "splunk",
                user_prompt=user_prompt
            )

        else:
            # Generate multiple queries from case title, description, task, and activity
            query_data = query_generator.generate_query_from_case(
                is_splunk=siem == "splunk",
                case_title=case_title,
                case_description=case_description,
                task_title=task_title,
                description=task_description,
                activity=activity,
            )

        if query_data is None:
            return Response(
                {"error": "Specified SIEM platform not implemented"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response({"result": query_data}, status=status.HTTP_200_OK)


class RestoreView(APIView):
    def post(self, request, *args, **kwargs):
        model_id = request.POST.get("model_id")

        if model_id is None:
            return Response(
                {"error": "No model specified"}, status=status.HTTP_400_BAD_REQUEST
            )
        if model_id not in config["models"].keys():
            return Response(
                {"error": "The model ID does not exist"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            trainer = QueryGenerationTrainer()
            trainer.load_baseline(model_id)

            return Response({"message": "Success"}, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError:
            raise OutOfMemoryError(
                "Ran out of GPU VRAM for query generation. Please make sure that your GPU has enough vram for the model."
            )
