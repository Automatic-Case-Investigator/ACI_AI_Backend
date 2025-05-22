from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

class APIKeyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        auth_header = request.headers.get('Authorization')
        
        response = self.get_response(request)
        
        if not auth_header or not auth_header.startswith('Bearer ') or auth_header[7:] != settings.API_KEY:
            response = Response({'error': 'Unauthorized'}, status=status.HTTP_401_UNAUTHORIZED)
            response.accepted_renderer = JSONRenderer()
            response.accepted_media_type = "application/json"
            response.renderer_context = {}
            response.render()
        
        return response