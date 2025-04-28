from task_generation_endpoint.objects.task_generation.task_generation_trainer import TaskGenerationTrainer
from task_generation_endpoint.objects.task_generation.task_generation_model import TaskGenerationModel
from task_generation_endpoint.objects.task_generation.task_generator import task_generator
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from ACI_AI_Backend.objects.redis_client import redis_client
from django.core.paginator import Paginator, EmptyPage
from rest_framework.response import Response
from task_generation_endpoint.models import *
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

file = open(settings.TASK_GENERATION_CONFIG_PATH, "r")
config = json.load(file)
file.close()
            
class TaskGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        case_title = request.POST.get("case_title")
        case_description = request.POST.get("case_description")
        if case_title is None or case_description is None:
            return Response({"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST)
        
        task_data = task_generator.generate_task(title=case_title, description=case_description)
        return Response({"result": task_data}, status=status.HTTP_200_OK)

class CaseTemporaryStorageView(APIView):
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
            
            formatted_data = dict()
            formatted_data["title"] = request_data["title"]
            formatted_data["description"] = request_data["description"]
            formatted_data["tasks"] = []
            for task in request_data["tasks"]:
                formatted_data["tasks"].append({
                    "title": task["title"],
                    "description": task["description"]
                })
            
            id_hash = hashlib.sha256(request_data["id"].encode()).hexdigest()
            title_hash = hashlib.sha256(formatted_data["title"].encode()).hexdigest()
            description_hash = hashlib.sha256(formatted_data["description"].encode()).hexdigest()
            tasks_hash = hashlib.sha256(str(formatted_data["tasks"]).encode()).hexdigest()
            
            id = f"Case:{id_hash + title_hash + description_hash + tasks_hash}"
            key_prefix = f"Case:{id_hash + title_hash + description_hash}*"
            
            for key in redis_client.scan_iter(match=key_prefix):
                redis_client.delete(key)
                
            redis_client.set(id, json.dumps(formatted_data), ex=settings.REDIS_KEY_EXPIRY_TIME)
            return Response({"message": "Success", "id": id}, status=status.HTTP_200_OK)
        except (KeyError, JSONDecodeError):
            return Response({"error": "Data not formatted properly"}, status=status.HTTP_400_BAD_REQUEST)
        

    def delete(self, request, *args, **kwargs):
        id = request.data.get("id")
        if id is None:
            return Response({"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST)

        redis_client.delete(id)
        return Response({"message": "Success"}, status=status.HTTP_200_OK)

class TaskGenTrainerView(APIView):
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
    
class RestoreView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            trainer = TaskGenerationTrainer()
            trainer.load_baseline()
            TaskGenerationModel.load()
            
            try:
                current_backup_model, _ = CurrentBackupModelEntry.objects.get_or_create(id=1)
                current_backup_model.current_model = None
                current_backup_model.save()
            except CurrentBackupModelEntry.DoesNotExist:
                pass
            return Response({"message": "Success"}, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError:
            raise OutOfMemoryError("Ran out of GPU VRAM for query generation. Please make sure that your GPU has enough vram for the model.")
        
class BackupView(APIView):
    def post(self, request):
        try:
            trainer = TaskGenerationTrainer()
            file_name = trainer.backup_model()
            name = os.path.basename(file_name).split('/')[-1].replace(".zip", "")
            backup_model_entry = BackupModelEntry(name=name, file_name=file_name)
            current_backup_model = CurrentBackupModelEntry(current_model=backup_model_entry)
            backup_model_entry.save()
            current_backup_model.save()
        
            return Response({"message": "Success"}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    def delete(self, request):
        try:          
            delete_name = request.data.get("hash")
            entry = BackupModelEntry.objects.get(name=delete_name)
            subprocess.run(["rm", entry.file_name])         
            entry.delete()
            
            return Response({"message": "Success"}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class RollbackView(APIView):
    def post(self, request):
        try:
            restore_name = request.POST.get("hash")
            backup = BackupModelEntry.objects.get(name=restore_name)
            subprocess.run(["unzip", "-o", backup.file_name])
       
            backup_name = os.path.basename(backup.file_name).split('/')[-1].replace(".zip", "")
            current_backup_model, _ = CurrentBackupModelEntry.objects.get_or_create(id=1)
            current_backup_model.current_model = backup
            current_backup_model.save()  
        
            return Response({"message": "Success"}, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class HistoryView(APIView):
    def get(self, request):
        try:
            page_number = int(request.GET.get("page"))
            all_backups = BackupModelEntry.objects.all().order_by("-date_created")
            paginator = Paginator(all_backups, 10)
            page_object = paginator.page(page_number)
            output = {"message" : "Success", "total_count": BackupModelEntry.objects.count(), "entries": []}
            
            for backup_entry in page_object.object_list:
                output["entries"].append({
                    "name": backup_entry.name,
                    "date_created": backup_entry.date_created
                })

            return Response(output, status=status.HTTP_200_OK)
        except EmptyPage:
            return Response({"message" : "Success", "total_count": BackupModelEntry.objects.count(), "entries": []}, status=status.HTTP_200_OK)
        except:
            print(traceback.format_exc())
            return Response({"error": "Data not formatted properly"}, status=status.HTTP_400_BAD_REQUEST)

class CurrentBackupVersionView(APIView):
    def get(self, request):
        try:
            current_backup_model, created = CurrentBackupModelEntry.objects.get_or_create(id=1)
            
            if created and current_backup_model.current_model is not None:
                return Response({"message" : "Success", "name": ""}, status=status.HTTP_200_OK)
            elif current_backup_model.current_model is not None:
                return Response({"message" : "Success", "name": current_backup_model.current_model.name}, status=status.HTTP_200_OK)
            else:
                return Response({"error" : "Failed to create backup entry"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except CurrentBackupModelEntry.DoesNotExist:
            return Response({"message" : "Success", "name": ""}, status=status.HTTP_200_OK)
        except EmptyPage:
            return Response({"message" : "Success", "total_count": BackupModelEntry.objects.count(), "entries": []}, status=status.HTTP_200_OK)
        except:
            print(traceback.format_exc())
            return Response({"error": "Data not formatted properly"}, status=status.HTTP_400_BAD_REQUEST)