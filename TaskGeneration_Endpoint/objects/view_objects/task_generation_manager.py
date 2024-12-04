from TaskGeneration_Endpoint.objects.task_generation.task_generation_trainer import TaskGenerationTrainer
from TaskGeneration_Endpoint.objects.task_generation.task_generator import task_generator
from ACI_AI_Backend.objects.redis_client import redis_client
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from django.conf import settings
from json import JSONDecodeError
import requests
import json
import uuid

class TaskGenerationManager(APIView):
    def post(self, request, *args, **kwargs):
        case_title = request.POST.get("case_title")
        case_description = request.POST.get("case_description")
        if case_title is None or case_description is None:
            return Response({"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST)
        
        task_data = task_generator.generate_task(title=case_title, description=case_description)
        return Response({"result": task_data}, status=status.HTTP_200_OK)

class CaseTemporaryStorageManager(APIView):
    def post(self, request, *args, **kwargs):
        """Modifies the already stored case data in redis

        JSON body format:
            ```json
            {
                "id": "...",
                "title": "...",
                "description": "...",
                "tasks" : [
                    {
                        "title": "...",
                        "description: "..."
                    },
                    ...
                ]
            }
            ```
            
        """
        try:
            request_data = json.loads(request.body)
            if "id" not in request_data.keys():
                return Response({"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST)
            
                    
            formatted_data = dict()
            formatted_data["title"] = request_data["title"]
            formatted_data["description"] = request_data["description"]
            formatted_data["tasks"] = []
            for task in request_data["tasks"]:
                formatted_data["tasks"].append({
                    "title": task["title"],
                    "description": task["description"]
                })
            
            id = f"Case:{str(uuid.uuid4())}"
            redis_client.set(id, json.dumps(formatted_data), ex=settings.REDIS_KEY_EXPIRY_TIME)
            return Response({"message": "Success", "id": id}, status=status.HTTP_200_OK)
        except (KeyError, JSONDecodeError):
            return JsonResponse({"error": "Data not formatted properly"}, status=status.HTTP_400_BAD_REQUEST)
        

    def delete(self, request, *args, **kwargs):
        id = request.data.get("id")
        if id is None:
            return Response({"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST)

        redis_client.delete(id)
        return Response({"message": "Success"}, status=status.HTTP_200_OK)

class TaskGenTrainerManager(APIView):
    def post(self, request, *args, **kwargs):
        try:
            seed = int(request.POST.get("seed"))
            max_steps = int(request.POST.get("max_steps"))
            learning_rate = float(request.POST.get("learning_rate"))
            gradient_accumulation_steps = int(request.POST.get("gradient_accumulation_steps"))
            weight_decay = float(request.POST.get("weight_decay"))
            
            trainer = TaskGenerationTrainer()
            trainer.load_dataset()
            trainer.load_model_tokenizer_locally()
            trainer.train(
                seed=seed,
                max_steps=max_steps,
                learning_rate=learning_rate,
                gradient_accumulation_steps=gradient_accumulation_steps,
                weight_decay=weight_decay
            )
            task_generator.__init__()
            return Response({"message": "Success"}, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except TypeError:
            return Response({"error": "Required fields have invalid format"}, status=status.HTTP_400_BAD_REQUEST)
    
class RestoreManager(APIView):
    def post(self, request, *args, **kwargs):
        try:
            trainer = TaskGenerationTrainer()
            trainer.load_baseline()
            task_generator.__init__()
            return Response({"message": "Success"}, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)