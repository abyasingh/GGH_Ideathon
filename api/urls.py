from django.urls import path
from . import views

urlpatterns = [
    path('api/water_pollution/', views.WaterPotabilityPredictionAPIView.as_view(), name='predict_water_potability'),
    path('api/climate_change/', views.ClimateChangePredictionAPIView.as_view(), name='predict_climate_change'),
    path('api/deforestation/', views.DeforestationPredictionAPIView.as_view(), name='predict_deforestation'),
    path('', views.homepage, name='homepage'),
    path('air_box/', views.air, name='air_box'),
    path('water_pollution/', views.water, name='water_pollution'),
    path('deforestation_rate/', views.deforestation, name='deforestation_rate'),
    path('climate_change/', views.climate, name='climate_change'),
]
