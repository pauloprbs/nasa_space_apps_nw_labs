from django.shortcuts import render

from rest_framework import generics
from .models import Measurement
from .serializers import MeasurementSerializer

class MeasurementList (generics.ListCreateAPIView):
    queryset = Measurement.objects.all()
    serializer_class = MeasurementSerializer

class MeasurementDetail (generics.RetrieveUpdateDestroyAPIView):
    queryset = Measurement.objects.all()
    serializer_class = MeasurementSerializer

# Create your views here.

