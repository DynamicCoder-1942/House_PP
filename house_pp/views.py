from django.http import HttpResponse
from django.shortcuts import render
import pickle
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler



def index(request):
    return render(request, 'index.html')

def predictions(request):
    a = request.GET.get('posted_by')
    b = request.GET.get('under_constuction')
    c = request.GET.get('rera_registered')
    d = request.GET.get('num_of_rooms')
    e = request.GET.get('bhk_or_rk')
    f = request.GET.get('sqaure_ft')
    g = request.GET.get('ready_to_move')
    h = request.GET.get('resale')
    i = request.GET.get('longitude')
    j = request.GET.get('latitude')
    final_model = joblib.load('django_model.pkl')

    scaler = MinMaxScaler()
    arr = np.array([[a, b, c, d, e, f, g, h, i, j]])
    b = scaler.fit_transform(arr)
    
    c = final_model.predict(b)
    answer = float(c*10000)
    z = round(answer, 2)
    



    params = {'result': z}
    return render(request,'result.html', params)