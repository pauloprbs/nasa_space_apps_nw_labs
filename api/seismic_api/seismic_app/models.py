from django.db import models

# Create your models here.

class Measurement(models.Model):
    mseed = models.CharField(max_length=200)
    time_abs = models.DateTimeField
    time_rel = models.FloatField
    velocity = models.FloatField

    def __str__(self):
        return self.mseed