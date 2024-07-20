from .views import RegistrationView
from .views import EmailVerificationView
from .views import UsernameVerificationView
from .views import LoginAccountView
from .views import LogoutView
from .views import ResetView
from .views import SetNewPasswordView
from django.urls import path
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    path('register',RegistrationView.as_view(),name="register"),
    path('validate-username', csrf_exempt(UsernameVerificationView.as_view()),name="validate-username"),   
    path('validate-email', csrf_exempt(EmailVerificationView.as_view()),name="validate-email"),
    path('login',LoginAccountView.as_view(),name="login"),
    path('resetpassword',ResetView.as_view(),name="resetpassword"),
    path('setnewpassword',SetNewPasswordView.as_view(),name="setnewpassword"),
    path('logout',LogoutView.as_view(),name="logout")
]
