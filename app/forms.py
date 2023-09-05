from django import forms
from .models import Audio

class AudioForm(forms.ModelForm):
    class Meta:
        model = Audio
        fields = ['title', 'audio_file']