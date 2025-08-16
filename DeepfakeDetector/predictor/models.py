from UserInfo.models import UploadHistory

# from django.db import models
# from django.contrib.auth.models import User
#
#
# class UploadHistory(models.Model):
#     MEDIA_TYPES = [
#         ('image', 'Image'),
#         ('video', 'Video'),
#         ('audio', 'Audio'),
#     ]
#
#     user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
#     file = models.FileField(upload_to='uploads/')
#     media_type = models.CharField(max_length=10, choices=MEDIA_TYPES)
#     prediction = models.CharField(max_length=50)  # e.g. Deepfake / Real
#     confidence = models.FloatField(null=True, blank=True)
#     uploaded_at = models.DateTimeField(auto_now_add=True)
#
#     def __str__(self):
#         return f"{self.media_type} - {self.prediction} ({self.uploaded_at})"
