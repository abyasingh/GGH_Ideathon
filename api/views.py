from rest_framework.views import APIView
from rest_framework.response import Response
import pickle
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from pathlib import Path
import xgboost as xgb
import numpy as np
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def homepage(request):
    return render(request, "homep.html")

def air(request):
    return render(request, "air_box.html")

def water(request):
    return render(request, "water_pollution.html")

def deforestation(request):
    return render(request, "deforestation_rate.html")

def climate(request):
    return render(request, "climate_change.html")

BASE_DIR = Path(__file__).resolve().parent.parent

class WaterPotabilityPredictionAPIView(APIView):
    def post(self, request, format=None):
        data = request.data
        ph = float(data.get('ph'))
        hardness = float(data.get('Hardness'))
        solids = float(data.get('Solids'))
        chloramines = float(data.get('Chloramines'))
        sulfate = float(data.get('Sulfate'))
        conductivity = float(data.get('Conductivity'))
        organic_carbon = float(data.get('Organic_carbon'))
        trihalomethanes = float(data.get('Trihalomethanes'))
        turbidity = float(data.get('Turbidity'))

        file_path = os.path.join(BASE_DIR, 'models/xgboost_model.pkl')
        classifier = pickle.load(open(file_path, 'rb'))
        feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        user_data = np.array([ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity])
        input_data = xgb.DMatrix(user_data.reshape(1, -1), feature_names=feature_names)
        prediction = classifier.predict(input_data)[0]
        water_potability = "The water is potable and fit for consumption." if prediction == 1 else "The water is not potable. Consider boiling or filtering before consumption."
        return Response({'water_potability': water_potability})

class ClimateChangePredictionAPIView(APIView):
    def post(self, request, format=None):
        data = request.data
        LandMaxTemperature = float(data.get('LandMaxTemperature'))
        LandMinTemperature = float(data.get('LandMinTemperature'))
        LandAndOceanAverageTemperature = float(data.get('LandAndOceanAverageTemperature'))
        booster = xgb.Booster()
        booster.load_model('models/climate_change.model')
        feature_names = ['LandMaxTemperature', 'LandMinTemperature', 'LandAndOceanAverageTemperature']
        user_data = np.array([LandMaxTemperature, LandMinTemperature, LandAndOceanAverageTemperature])
        input_data = xgb.DMatrix(user_data.reshape(1, -1), feature_names=feature_names)
        prediction = booster.predict(input_data)[0]
        return Response({'prediction': prediction})
    
class DeforestationPredictionAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        image = request.FILES.get('image')
        model_path = os.path.join(settings.BASE_DIR, 'models/deforestation.h5')
        tmp_file = default_storage.save('tmp.jpg', ContentFile(image.read()))
        image_path = os.path.join(settings.BASE_DIR, tmp_file)
        classifier = load_model(model_path)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = classifier.predict(img)
        predicted_class = np.argmax(prediction)
        predicted_class = int(predicted_class) 
        default_storage.delete(tmp_file)
        return JsonResponse({'prediction': predicted_class})