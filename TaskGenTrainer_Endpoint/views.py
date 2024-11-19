from ACI_AI_Backend.objects.redis_client import redis_client
from django.http import JsonResponse
from json import JSONDecodeError
import json
import uuid

def add_case_data(request):
    """Stores case data temporarily in redis

    POST JSON body format:
        ```json
        {
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
            formatted_data = dict()
            formatted_data["title"] = request_data["title"]
            formatted_data["description"] = request_data["description"]
            formatted_data["tasks"] = []
            for task in request_data["tasks"]:
                formatted_data["tasks"].append({
                    "title": task["title"],
                    "description": task["description"]
                })
            
            id = f"Case-{str(uuid.uuid4())}"
            redis_client.set(id, json.dumps(formatted_data))
            return JsonResponse({"message": "Success", "id": id})
        except (KeyError, JSONDecodeError):
            return JsonResponse({"error": "Data not formatted properly"})
    
    else:
        return JsonResponse({"error": "Invalid method"})
    
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
            
            redis_client.set(id, json.dumps(formatted_data))
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
    pass