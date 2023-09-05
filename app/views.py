from django.http import HttpResponse
from django.http.response import StreamingHttpResponse
from django.shortcuts import render, redirect
from .forms import AudioForm

import pandas as pd
import matplotlib.pyplot as plt
# from app.diarizer import ___
import numpy as np
df = pd.DataFrame()
# Create your views here.

# file_name = 'CustomerCare'

def landing(request):
    return render(request, "app/landing.html")


def index(request):
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES or None)
        if form.is_valid():
            form.save()
            response = redirect('/index')
            return response
    else:
        form = AudioForm()
    return render(request, 'app/index.html', {'form': form})    


def audio_breakdown(request):
    from .emotion_detection import create_data_frame
    df = create_data_frame()
    data_list = df.values.tolist()
    print(data_list)
    context = {'data_list': data_list}
    return render(request, 'app/voice_breakdown.html', context)

def detection(request):
    from .emotion_detection import create_data_frame, output
    df = create_data_frame()
    outputs = output()
    emotion = []
    for i in range(len(outputs)):
        max_emotion = max(outputs[i], key=lambda x: float(x['Score'].strip('%')))
        emotion.append(max_emotion["Emotion"])

    df["Emotion"] = emotion
    data_list = df.values.tolist()
    print(data_list)
    context = {'data_list': data_list}
    return render(request, 'app/detection.html', context)