from django import forms

class MediaUploadForm(forms.Form):
    file = forms.FileField(label='Select an Image, Video, or Audio file')