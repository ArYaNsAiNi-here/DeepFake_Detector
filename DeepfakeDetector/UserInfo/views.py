from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from .models import UploadHistory

def dashboard(request):
    if not request.user.is_authenticated:
        return redirect('login')

    history = UploadHistory.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, "UserInfo/dashboard.html", {"history": history})


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid username or password")

    return render(request, 'UserInfo/login.html')


def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
        else:
            user = User.objects.create_user(username=username, password=password)
            login(request, user)  # Auto login after registration
            return redirect('dashboard')

    return render(request, 'UserInfo/register.html')


def logout_view(request):
    logout(request)
    return redirect('login')
