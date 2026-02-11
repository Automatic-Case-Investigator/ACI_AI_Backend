from task_generation_endpoint.objects.task_generation.task_generation_agent import task_generation_agent
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher
from rest_framework.response import Response
from rest_framework.views import APIView
from ACI_AI_Backend.llmtool import Tool
from rest_framework import status

class TaskGenerationView(APIView):
    """
    The view that handles task generation requests
    """
    def post(self, request, *args, **kwargs):
        case_title = request.POST.get("case_title")
        case_description = request.POST.get("case_description")
        web_search_enabled = request.POST.get("web_search")

        if case_title is None or case_description is None:
            return Response(
                {"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST
            )
        
        if web_search_enabled is None:
            web_search_enabled = False
        elif not web_search_enabled.isdigit():
            return Response(
                {"error": 'Parameter "web_search" not formatted properly'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        else:
            web_search_enabled = bool(int(web_search_enabled))

        context = None
        if web_search_enabled:
            searcher = WebSearcher()
            context = searcher.run(f"Case Title: {case_title}\n\nCase Description: {case_description}\n")

        task_data = task_generation_agent.invoke(case_title=case_title, case_description=case_description, web_search_context=context)
        return Response({"result": task_data}, status=status.HTTP_200_OK)
        