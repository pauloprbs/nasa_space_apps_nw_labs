from django.urls import path
from .views import MeasurementList, MeasurementDetail

urlpatterns = [
    path('measurement/', MeasurementList.as_view(), name='measurement-list'),
    path('measurement/<int:pk>/', MeasurementDetail.as_view(), name='measurement-detail'),
]