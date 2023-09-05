from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("", views.landing, name='landing'),
    path("index", views.index, name='index'),
    # path('upload-audio/', views.upload_audio, name='upload_audio'),
    path('audio_breakdown/', views.audio_breakdown, name='audio_breakdown'),
    path('detection/', views.detection, name='detection'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

