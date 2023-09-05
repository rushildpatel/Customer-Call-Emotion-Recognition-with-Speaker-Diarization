from django.db import models

# Create your models here.
class Audio(models.Model):
    title = models.CharField(max_length=255)
    audio_file = models.FileField(upload_to='audio/')

    class Meta:
        db_table = 'Audio'