from activity_generation_endpoint.objects.activity_generation.activity_generation_trainer import ActivityGenerationTrainer
from activity_generation_endpoint.objects.activity_generation.activity_generation_model import ActivityGenerationModel
from activity_generation_endpoint.objects.activity_generation.activity_generator import activity_generator
from ACI_AI_Backend.objects.redis_client import redis_client
from django.core.paginator import Paginator, EmptyPage
from rest_framework.response import Response
from activity_generation_endpoint.models import *
from rest_framework.views import APIView
from rest_framework import status
from django.conf import settings
from json import JSONDecodeError
import subprocess
import traceback
import requests
import hashlib
import json
import os

file = open(settings.ACTIVITY_GENERATION_CONFIG_PATH, "r")
config = json.load(file)
file.close()
            
class ActivityGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        case_title = request.POST.get("case_title")
        case_description = request.POST.get("case_description")
        task_title = request.POST.get("task_title")
        task_description = request.POST.get("task_description")
        if task_title is None or task_description is None:
            return Response({"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST)
        
        activity_data = activity_generator.generate_activity(
            case_title=case_title,
            case_description=case_description,
            task_title=task_title,
            description=task_description
        )
        return Response({"result": activity_data}, status=status.HTTP_200_OK)

class RestoreView(APIView):
    def post(self, request, *args, **kwargs):
        model_id = request.POST.get("model_id")
        
        if model_id is None:
            return Response({"error": "No model specified"}, status=status.HTTP_400_BAD_REQUEST)
        if model_id not in config["models"].keys():
            return Response({"error": "The model ID does not exist"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            trainer = ActivityGenerationTrainer()
            trainer.load_baseline(model_id)
            
            return Response({"message": "Success"}, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)