from query_generation_endpoint.objects.query_generation.query_generation_agent import query_generation_agent
from ACI_AI_Backend.objects.web_search.web_searcher import WebSearcher
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

class QueryGenerationView(APIView):
    def post(self, request, *args, **kwargs):
        user_prompt = request.POST.get("prompt")
        case_title = request.POST.get("case_title")
        case_description = request.POST.get("case_description")
        task_title = request.POST.get("task_title")
        task_description = request.POST.get("task_description")
        activity = request.POST.get("activity")
        siem = request.POST.get("siem")
        web_search_enabled = request.POST.get("web_search")

        if case_title is None or case_description is None:
            return Response(
                {"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST
            )
        
        if task_title is None or task_description is None:
            return Response(
                {"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST
            )
        
        if activity is None:
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


        query_data = None
        sources = set()
        if user_prompt:
            context = None
            if web_search_enabled:
                searcher = WebSearcher()
                context = searcher.run(user_prompt)

                # Aggregate sources for each keyword
                for keyword in context.keys():
                    sources.update(context[keyword]["sources"])

            # Defaults to generate from prompt if it is provided
            query_data = query_generation_agent.invoke(
                is_splunk=siem == "splunk",
                user_prompt=user_prompt,
                web_search_context=context
            )

        else:
            context = None
            if web_search_enabled:
                searcher = WebSearcher()
                context = searcher.run(f"Case Title: {case_title}\n\nCase Description: {case_description}\n\nTask Title: {task_title}\n\nTask Description: {task_description}\n")
                
                # Aggregate sources for each keyword
                for keyword in context.keys():
                    sources.update(context[keyword]["sources"])

            # Generate multiple queries from case title, description, task, and activity
            query_data = query_generation_agent.invoke(
                is_splunk=siem == "splunk",
                case_title=case_title,
                case_description=case_description,
                task_title=task_title,
                description=task_description,
                activity=activity,
                web_search_context=context
            )


        if query_data is None:
            return Response(
                {"error": "Specified SIEM platform not implemented"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response({"result": query_data, "sources": list(sources)}, status=status.HTTP_200_OK)
