from activity_generation_endpoint.objects.activity_generation.activity_generation_agent import activity_generation_agent
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher
from rest_framework.response import Response
from rest_framework.views import APIView
from ACI_AI_Backend.llmtool import Tool
from rest_framework import status

class ActivityGenerationView(APIView):
    """
    The view that handles activity generation requests
    """
    def post(self, request, *args, **kwargs):
        case_title = request.POST.get("case_title")
        case_description = request.POST.get("case_description")
        task_title = request.POST.get("task_title")
        task_description = request.POST.get("task_description")
        web_search_enabled = request.POST.get("web_search")

        if case_title is None or case_description is None:
            return Response(
                {"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST
            )
        
        if task_title is None or task_description is None:
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
            context = searcher.run(f"Case Title: {case_title}\n\nCase Description: {case_description}\n\nTask Title: {task_title}\n\nTask Description: {task_description}\n")

        activity_data = activity_generation_agent.invoke(case_title=case_title, case_description=case_description, task_title=task_title, task_description=task_description, web_search_context=context)
        return Response({"result": activity_data}, status=status.HTTP_200_OK)
        