from django.urls import path
from . import views

urlpatterns = [
    # The root URL (e.g., /) now serves the main page
    path('', views.upload_page, name='upload_page'),

    # The /detect URL is the API endpoint for file processing
    path('detect/', views.detect_api, name='detect_api'),
]