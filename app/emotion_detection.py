import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd
import soundfile

from transformers import AutoConfig, Wav2Vec2FeatureExtractor

import sys
sys.path.insert(0,'D:/SPIT/8th Sem/Major Project/SpeakerDiarization/soxan/src')

from models import Wav2Vec2ClassificationHead, Wav2Vec2ForSpeechClassification
# Load the .wav file
audio_file = 'D:/SPIT/8th Sem/Major Project/SpeakerDiarization/audio/test_bro_new.wav'
audio_data, sr = soundfile.read(audio_file)

# Define the start and stop times in seconds

def create_data_frame():
    with open('D:/SPIT/8th Sem/Major Project/SpeakerDiarization/audio/rttm/test_bro_new.rttm') as f:
        lines = f.readlines()
        lines = [i.split(' ') for i in lines]

        start, speaker, end = [], [], []
        for idx, line in enumerate(lines):
            start.append(round(float(line[3]),2))
            end.append(round(start[idx] + float(line[4]),2))
            speaker.append(round(int(line[7][-2:]),0))

        df = pd.DataFrame({'Start': start, 'End': end, 'Speaker': speaker})
        return df
        # df = df[df['Speaker'] == 1]

def create_chunks(df):
    print(df)
    for i in df.index:
        start_time = df["Start"][i]
        stop_time = df["End"][i]
        
        # Convert the start and stop times from seconds to samples
        start_sample = int(start_time * sr)
        stop_sample = int(stop_time * sr)

        # Extract the audio segment between the start and stop times
        audio_segment = audio_data[start_sample:stop_sample]

        # Save the audio segment to a new file
        segment_file = 'D:/SPIT/8th Sem/Major Project/SpeakerDiarization/audio/segments/test_bro_new_{}.wav'.format(i)
        soundfile.write(segment_file, audio_segment, sr)

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict(path, sampling_rate, device, config, feature_extractor, model):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs

def output():
    # path for a sample
    outputs = [] 
    df = pd.DataFrame()
    df = create_data_frame()
    create_chunks(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name_or_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
    config = AutoConfig.from_pretrained(model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    sampling_rate = feature_extractor.sampling_rate
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
    model.gradient_checkpointing_enable()

    for i in df.index:
        path = 'D:/SPIT/8th Sem/Major Project/SpeakerDiarization/audio/segments/test_bro_new_{}.wav'.format(i)   
        outputs.append(predict(path, sampling_rate, device, config, feature_extractor, model))

    for i in df.index:
        print(outputs[i])

    return outputs