import json
import firebase_admin
from django.shortcuts import render, redirect
from django.views import View
from django.http import JsonResponse
from django.contrib.auth.models import User
from validate_email import validate_email
from django.contrib import messages
from django.urls import reverse
from django.contrib import auth
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from firebase_admin import auth as firebase_auth
from django.contrib.auth import authenticate, login
from firebase_admin import firestore
from django.contrib.auth.hashers import make_password
from firebase_admin import auth as firebase_auth
import pyrebase

from django.contrib.auth import logout
#from django.core.mail import EmailMessage
#from django.utils.encoding import force_bytes, force_str, DjangoUnicodeDecodeError
#from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
#from django.contrib.sites.shortcuts import get_current_site
#from .utils import account_activation_token

# Firebase Initialization
if not firebase_admin._apps:
    print("Initializing Firebase application...")
    cred = firebase_admin.credentials.Certificate("soccer-webapp-firebase-adminsdk-zfjgb-e7fd33d3da.json")
    firebase_admin.initialize_app(cred)
    print("Firebase application initialized successfully.")
else:
    print("Firebase application is already initialized.")

config = {
  "apiKey": "AIzaSyDNopQcimASaKlzujW4s_oYAp5JoKLY_yg",
  "authDomain": "soccer-webapp.firebaseapp.com",
  "projectId": "soccer-webapp",
  "storageBucket": "soccer-webapp.appspot.com",
  "databaseURL": "",
  "messagingSenderId": "496623562168",
  "appId": "1:496623562168:web:ce6e977030680d6481334d",
}
firebase = pyrebase.initialize_app(config)

def index(request):
    return render(request, 'webapp/index.html')

class RegistrationView(View):
    def get(self, request):
        return render(request, 'authentication/register.html')

    def post(self, request):
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        context = {'fieldValues': request.POST}

        if not username or not email or not password:
            messages.error(request, 'Please fill in all the fields')
            return render(request, 'authentication/register.html', context)

        if len(password) < 8:
            messages.error(request, 'Password must be at least 8 characters long and contain letters, numbers, and special characters.')
            return render(request, 'authentication/register.html', context)

        try:
            # Check if the user already exists in Firebase
            firebase_auth.get_user_by_email(email)
            messages.error(request, 'Email address already registered')
            return render(request, 'authentication/register.html', context)
        except firebase_auth.UserNotFoundError:
            pass

        try:
            # Create the user in Firebase and set them as verified
            user = firebase_auth.create_user(
                email=email,
                password=password,
                display_name=username,
                email_verified=True)
            # Create the user in Django
            django_user = User.objects.create_user(username=username, email=email, password=password)
            login(request, django_user)
            messages.success(request, 'Account successfully created. You are now logged in.')
            return redirect('login')
        
        except Exception as e:
            messages.error(request, f'Error creating user: {str(e)}')
            return render(request, 'authentication/register.html', context)

class EmailVerificationView(View):
    def post(self, request):
        data = json.loads(request.body)
        email = data['email']

        # Check if the email is valid
        if not validate_email(email):
            return JsonResponse({'email_error': 'Email is invalid'}, status=400)        
        # Check if the email already exists in the database
        if User.objects.filter(email=email).exists():
            return JsonResponse({'email_error': 'Email already in use'}, status=409)
        return JsonResponse({'email_valid': True})

class UsernameVerificationView(View):
    def post(self, request):
        data = json.loads(request.body)
        username = data['username']
        # Check if the username is alphanumeric
        if not str(username).isalnum():
            return JsonResponse({'username_error': 'Username should only contain alphanumeric characters'}, status=400)
        # Check if the username already exists in the database
        if User.objects.filter(username=username).exists():
            return JsonResponse({'username_error': 'Username already in use'}, status=409)
        return JsonResponse({'username_valid': True})
    
class LoginAccountView(View):
    def get(self, request):
        next_url = request.GET.get('next')
        return render(request, 'authentication/login.html', {'next': next_url})

    def post(self, request):
        email = request.POST.get('email')
        password = request.POST.get('password')
        auther = firebase.auth()
        username = User.objects.get(email=email.lower()).username

        if not email or not password:
            messages.error(request, 'Please fill all fields')
            return render(request, 'authentication/login.html')
        # Try Firebase authentication
        try:
            user = auther.sign_in_with_email_and_password(email, password)
            print("User signed in successfully with Firebase")
            userd = authenticate(username=username, password=password)
            login(request, userd)
            print("User signed in successfully with django also")
            messages.success(request, f'Welcome, {user["email"]} you are now logged in')
            messages.success(request, 'Logged in via Firebase and Django')
            return redirect(reverse('index') + f'?username={username}')
        except Exception as e:
            print(f"Firebase login failed: {e}")
        return render(request, 'authentication/login.html')

class ResetView(View):
    def get(self,request):
        return render(request, "authentication/resetpassword.html")


class SetNewPasswordView(View):
    def get(self,request):
        return render(request, "authentication/resetpassword.html")
    
    def post(self,request):
        auther = firebase.auth()
        email = request.POST.get('email')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')
        try:
            user = firebase_auth.get_user_by_email(email)
            firebase_auth.update_user(user.uid, password=new_password)
            #auther.send_password_reset_email(email)
            #message  = "A email to reset password is successfully sent"
            django_user = User.objects.get(email=email)
            django_user.password = make_password(new_password)
            django_user.save()      
            message = "Password successfully reset."
            return render(request, "authentication/setnewpassword.html", {"msg":message})
        except:
            message  = "Something went wrong, Please check the email you provided is registered or not"
            return render(request, "authentication/resetpassword.html", {"msg":message})


class LogoutView(View):
    def post(self, request):
        logout(request)
        messages.success(request, 'You have been logged out')
        return redirect('login')
