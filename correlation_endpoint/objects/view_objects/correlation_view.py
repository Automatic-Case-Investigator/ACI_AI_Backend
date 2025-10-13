from correlation_endpoint.objects.correlator.wazuh_correlator import WazuhCorrelator
from ACI_AI_Backend.objects.exceptions.out_of_memory_error import OutOfMemoryError
from correlation_endpoint.utils.preprocessing_fn import preprocess_wazuh_event
from ACI_AI_Backend.objects.device import get_freest_device
from rest_framework.response import Response
from query_generation_endpoint.models import *
from rest_framework.views import APIView
from rest_framework import status
import traceback


siem_classes = {"WZ": WazuhCorrelator}

siem_event_preprocessing_fn = {"WZ": preprocess_wazuh_event}


class CorrelationView(APIView):
    def post(self, request, *args, **kwargs):
        event_data = request.POST.get("event")
        case_title = request.POST.get("case_title")
        case_description = request.POST.get("case_description")
        activity = request.POST.get("activity")
        siem_type = request.POST.get("siem_type")

        if event_data is None:
            return Response({"error": "Required field missing"}, status=status.HTTP_400_BAD_REQUEST)

        predict_result = 0
        try:
            device = get_freest_device()
            anomaly_detector = siem_classes[siem_type](device=device)
            anomaly_detector.load_pretrained("correlation_endpoint/objects/correlator/trained_model/wazuh_correlator/model.pt")

            event_data = siem_event_preprocessing_fn[siem_type](event_data)
            predict_result = anomaly_detector.predict(event_data, case_title, case_description, activity)[0]
        except OutOfMemoryError as e:
            traceback.format_exc()
            return Response({"error": e}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"result": predict_result}, status=status.HTTP_200_OK)
