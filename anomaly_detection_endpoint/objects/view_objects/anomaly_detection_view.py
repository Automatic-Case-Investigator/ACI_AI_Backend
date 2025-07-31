from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from anomaly_detection_endpoint.objects.anomaly_detector.wazuh_anomaly_detector import WazuhAnomalyDetector
from ACI_AI_Backend.objects.device import DEVICE, update_current_device
from rest_framework.response import Response
from query_generation_endpoint.models import *
from rest_framework.views import APIView
from rest_framework import status
from django.conf import settings
import traceback

siem_classes = {
    "WZ": WazuhAnomalyDetector
}

class AnomalyDetectionView(APIView):
    def post(self, request, *args, **kwargs):
        event_data = request.POST.get("event_data")
        siem_type = request.POST.get("siem_type")

        if event_data is None:
            return Response(
                {"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST
            )
        
        predict_result = 0
        try:
            update_current_device()
            anomaly_detector = siem_classes[siem_type](device=DEVICE)
            anomaly_detector.load_pretrained("anomaly_detection_endpoint/objects/anomaly_detector/trained_model/wazuh_anomaly_detector/model.pt")

            predict_result = anomaly_detector.predict(event_data)[0]
        except OutOfMemoryError as e:
            traceback.format_exc()
            return Response({"error": e}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        
        return Response({"result": predict_result}, status=status.HTTP_200_OK)

