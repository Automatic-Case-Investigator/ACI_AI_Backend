from django.conf import settings
import socket

class ModelUpdateNotifier:
    @classmethod
    def notify_update(self, model_name):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((settings.NOTIFIER_HOST, int(settings.NOTIFIER_PORT)))
            client_socket.sendall(model_name.encode())


