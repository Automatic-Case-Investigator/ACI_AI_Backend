from TaskGenTrainer_Endpoint.objects.task_generation_trainer.task_generation_trainer import TaskGenerationTrainer
from ACI_AI_Backend.objects.redis_client import redis_client
from django.http import JsonResponse
from django.conf import settings
from json import JSONDecodeError
import json
import uuid

def set_case_data(request):
    """Modifies the already stored case data in redis

    POST JSON body format:
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
        

    Args:
        request (_type_): django request

    Returns:
        JsonResponse: Json response
    """
    if request.method == "POST":
        try:
            request_data = json.loads(request.body)
            if "id" not in request_data.keys():
                return JsonResponse({"error": "Required field missing"})
            
            id = request_data["id"]
        
            formatted_data = dict()
            formatted_data["title"] = request_data["title"]
            formatted_data["description"] = request_data["description"]
            formatted_data["tasks"] = []
            for task in request_data["tasks"]:
                formatted_data["tasks"].append({
                    "title": task["title"],
                    "description": task["description"]
                })
            
            redis_client.set(id, json.dumps(formatted_data), ex=settings.REDIS_KEY_EXPIRY_TIME)
            return JsonResponse({"message": "Success", "id": id})
        except (KeyError, JSONDecodeError):
            return JsonResponse({"error": "Data not formatted properly"})
    
    else:
        return JsonResponse({"error": "Invalid method"})
    
def delete_case_data(request):
    """Removes case data redis

    POST parameters:
        id: case data id

    Args:
        request (_type_): django request

    Returns:
        JsonResponse: Json response
    """
    if request.method == "POST":
        id = request.POST.get("id")
        if id is None:
            return JsonResponse({"error": "Required field missing"})

        redis_client.delete(id)
        return JsonResponse({"message": "Success"})
    
    else:
        return JsonResponse({"error": "Invalid method"})

def train_model(request):
    if request.method == "POST":
        try:
            trainer = TaskGenerationTrainer()
            trainer.load_dataset()
            trainer.load_model_tokenizer_locally()
            trainer.train()
            return JsonResponse({"message": "Success"})
        except ValueError as e:
            return JsonResponse({"error": str(e)})
    else:
        return JsonResponse({"error": "Invalid method"})

def restore_baseline(request):
    if request.method == "POST":
        try:
            trainer = TaskGenerationTrainer()
            trainer.load_baseline()
            return JsonResponse({"message": "Success"})
        except ValueError as e:
            return JsonResponse({"error": str(e)})
    else:
        return JsonResponse({"error": "Invalid method"})