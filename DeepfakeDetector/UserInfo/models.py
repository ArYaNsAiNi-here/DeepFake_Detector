from django.db import models
from django.contrib.auth.models import User

class UploadHistory(models.Model):
    MEDIA_TYPES = [
        ('image', 'Image'),
        ('video', 'Video'),
        ('audio', 'Audio'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/')
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPES)
    prediction = models.CharField(max_length=50)

    # Changed from "confidence" to "accuracy"
    accuracy = models.FloatField(null=True, blank=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        acc_display = f"{self.accuracy:.2f}%" if self.accuracy is not None else "N/A"
        return f"{self.user.username} - {self.media_type} - {self.prediction} ({acc_display})"
