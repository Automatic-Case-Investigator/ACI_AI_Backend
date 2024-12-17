from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
import requests

class TestConnection(APIView):
    def get(self, request, *args, **kwargs):
        return Response({"message": "Success", "connected": True}, status=status.HTTP_200_OK)