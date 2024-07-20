from django.urls import path
from . import views
from .views import team_data
from .views import generate_graphs
from .views import machine_learning
from .views import spain_view
from .views import italy_view
from .views import germany_view
from .views import france_view
from .views import england_view
from .views import mexico_view
from .views import usa_view
from .views import welcome
from .views import toggle_favorite
from .views import profile

#webapp urls.py
print("Configuring URLs for webapp module")
urlpatterns = [
    path('',views.index,name="index"),
    path('welcome/', welcome, name='welcome'),
    path('webapp/', views.index, name='webapp'),
    #path('webapp/',welcome,name="webapp"),
    path('team_data/', team_data, name='team_data'),
    path('data_insights/', generate_graphs, name='data_insights'),
    path('spain/',spain_view, name='spain'),
    path('italy/',italy_view, name='italy'),
    path('germany/',germany_view, name='germany'),
    path('france/',france_view, name='france'),
    path('england/',england_view, name='england'),
    path('mexico/',mexico_view, name='mexico'),
    path('usa/',usa_view, name='usa'),
    path('toggle_favorite/',toggle_favorite, name='toggle_favorite'),
    path('profile/', profile, name='profile'),
    path('machine_learning/', machine_learning, name='machine_learning'),
    path('knn/', views.knn_view, name='knn'),
    path('decision_tree/', views.decision_tree_view, name='decision_tree'),
    path('linear_regression/', views.linear_regression_view, name='linear_regression'),
    path('svm/', views.svm_view, name='svm'),
    path('summary/', views.summary_view, name='summary'),
]
