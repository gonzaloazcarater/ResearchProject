from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import TeamData, UserFavorite
from django.http import HttpResponse
import matplotlib.pyplot as plt
import seaborn as sns
from django.http import JsonResponse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from .forms import ToggleFavoriteForm

# Create your views here.
#webapp/views.py

def welcome(request):
    return render(request, 'webapp/welcome.html')

@login_required(login_url='/authentication/login')
def index(request):
    username = request.GET.get('username')
    context = {
    'username': username,
    }
    return render(request, 'webapp/index.html',context)


@login_required(login_url='/authentication/login')
def team_data(request):
    teams = TeamData.objects.all()
    user_favorites = UserFavorite.objects.filter(user=request.user).values_list('team_id', flat=True)
    return render(request, 'webapp/team_data.html', {'teams': teams, 'user_favorites': list(user_favorites)})

@login_required(login_url='/authentication/login')
@require_POST
def toggle_favorite(request):
    form = ToggleFavoriteForm(request.POST)
    if form.is_valid():
        success = form.toggle_favorite(request.user)
        if success:
            print("Favorite toggle processed successfully.")
        else:
            print("Error processing favorite toggle.")
        return JsonResponse({'success': success})
    else:
        print("Invalid form for favorite toggle.")
        return JsonResponse({'success': False, 'errors': form.errors}, status=400)


@login_required(login_url='/authentication/login')
def generate_graphs(request):
    return render(request,'webapp/data_insights.html')

@login_required(login_url='/authentication/login')
def machine_learning(request):
    if request.method == 'POST':
        selected_model = request.POST.get('model_select')
        return JsonResponse({'message': 'Modelo seleccionado: {}'.format(selected_model)})
    else:
        return render(request, 'webapp/machine_learning.html')

def scrape_spain_data():
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
    url = "https://www.transfermarkt.us/laliga/besucherzahlen/wettbewerb/ES1/saison_id/2023"
    response = requests.get(url, headers=headers)
    page_soup = BeautifulSoup(response.content, 'html.parser')    
    teams_html = page_soup.find_all('tr', class_='odd') + page_soup.find_all('tr', class_='even')
    
    teams_data = []
    for team_html in teams_html:
        cells = team_html.find_all('td')
        if len(cells) >= 6:  
            teamname = cells[4].text.strip()  
            stadium = cells[3].text.strip()
            capacity = int(float(cells[5].text.strip().replace(",", "").replace(".", "")))# Convert capacity and number of spectators to integers
            numofspecators = int(float(cells[6].text.strip().replace(",", "").replace(".", "")))
            average = int(float(cells[7].text.strip().replace(",", "").replace(".", ""))) # Convert average to float
            teams_data.append({
                'Team_Name': teamname,
                'Stadium': stadium,
                'Capacity': capacity,
                'Number_of_Spectators': numofspecators,
                'Average': average
            })
    return teams_data

@login_required(login_url='/authentication/login')
def spain_view(request):
    data = scrape_spain_data()
    #print(data)
    return render(request, 'webapp/spain.html', {'teams_data': data})

def scrape_italy_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'
    }
    url = "https://www.transfermarkt.us/serie-a/besucherzahlen/wettbewerb/IT1"
    
    response = requests.get(url, headers=headers)
    page_soup = BeautifulSoup(response.content, 'html.parser')
    
    teams_html = page_soup.find_all('tr', class_='odd') + page_soup.find_all('tr', class_='even')
    
    teams_data = []
    for team_html in teams_html:
        cells = team_html.find_all('td')
        
        if len(cells) >= 6:  
            teamname = cells[4].text.strip()  
            stadium = cells[3].text.strip()
            # Convert capacity and number of spectators to integers
            capacity = int(float(cells[5].text.strip().replace(",", "").replace(".", "")))
            numofspecators = int(float(cells[6].text.strip().replace(",", "").replace(".", "")))
            
            # Convert average to float
            average = int(float(cells[7].text.strip().replace(",", "").replace(".", ""))) 
            
            teams_data.append({
                'Team_Name': teamname,
                'Stadium': stadium,
                'Capacity': capacity,
                'Number_of_Spectators': numofspecators,
                'Average': average
            })
    return teams_data

@login_required(login_url='/authentication/login')
def italy_view(request):
    data = scrape_italy_data()
    #print(data)
    return render(request, 'webapp/italy.html', {'teams_data': data})

def scrape_france_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'
    }
    url = "https://www.transfermarkt.com/ligue-1/besucherzahlen/wettbewerb/FR1"
    
    response = requests.get(url, headers=headers)
    page_soup = BeautifulSoup(response.content, 'html.parser')
    
    teams_html = page_soup.find_all('tr', class_='odd') + page_soup.find_all('tr', class_='even')
    
    teams_data = []
    for team_html in teams_html:
        cells = team_html.find_all('td')
        
        if len(cells) >= 6:  
            teamname = cells[4].text.strip()  
            stadium = cells[3].text.strip()
            # Convert capacity and number of spectators to integers
            capacity = int(float(cells[5].text.strip().replace(",", "").replace(".", "")))
            numofspecators = int(float(cells[6].text.strip().replace(",", "").replace(".", "")))
            
            # Convert average to float
            average = int(float(cells[7].text.strip().replace(",", "").replace(".", ""))) 
            
            teams_data.append({
                'Team_Name': teamname,
                'Stadium': stadium,
                'Capacity': capacity,
                'Number_of_Spectators': numofspecators,
                'Average': average
            })
    return teams_data

@login_required(login_url='/authentication/login')
def france_view(request):
    data = scrape_france_data()
    #print(data)
    return render(request, 'webapp/france.html', {'teams_data': data})

def scrape_england_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'
    }
    url = "https://www.transfermarkt.us/premier-league/besucherzahlen/wettbewerb/GB1"
    
    response = requests.get(url, headers=headers)
    page_soup = BeautifulSoup(response.content, 'html.parser')
    
    teams_html = page_soup.find_all('tr', class_='odd') + page_soup.find_all('tr', class_='even')
    
    teams_data = []
    for team_html in teams_html:
        cells = team_html.find_all('td')
        
        if len(cells) >= 6:  
            teamname = cells[4].text.strip()  
            stadium = cells[3].text.strip()
            # Convert capacity and number of spectators to integers
            capacity = int(float(cells[5].text.strip().replace(",", "").replace(".", "")))
            numofspecators = int(float(cells[6].text.strip().replace(",", "").replace(".", "")))
            
            # Convert average to float
            average = int(float(cells[7].text.strip().replace(",", "").replace(".", ""))) 
            
            teams_data.append({
                'Team_Name': teamname,
                'Stadium': stadium,
                'Capacity': capacity,
                'Number_of_Spectators': numofspecators,
                'Average': average
            })
    return teams_data

@login_required(login_url='/authentication/login')
def england_view(request):
    data = scrape_england_data()
    #print(data)
    return render(request, 'webapp/england.html', {'teams_data': data})

def scrape_germany_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'
    }
    url = "https://www.transfermarkt.com/1-bundesliga/besucherzahlen/wettbewerb/L1"
    
    response = requests.get(url, headers=headers)
    page_soup = BeautifulSoup(response.content, 'html.parser')
    
    teams_html = page_soup.find_all('tr', class_='odd') + page_soup.find_all('tr', class_='even')
    
    teams_data = []
    for team_html in teams_html:
        cells = team_html.find_all('td')
        
        if len(cells) >= 6:  
            teamname = cells[4].text.strip()  
            stadium = cells[3].text.strip()
            # Convert capacity and number of spectators to integers
            capacity = int(float(cells[5].text.strip().replace(",", "").replace(".", "")))
            numofspecators = int(float(cells[6].text.strip().replace(",", "").replace(".", "")))
            
            # Convert average to float
            average = int(float(cells[7].text.strip().replace(",", "").replace(".", ""))) 
            
            teams_data.append({
                'Team_Name': teamname,
                'Stadium': stadium,
                'Capacity': capacity,
                'Number_of_Spectators': numofspecators,
                'Average': average
            })
    return teams_data

@login_required(login_url='/authentication/login')
def germany_view(request):
    data = scrape_germany_data()
    #print(data)
    return render(request, 'webapp/germany.html', {'teams_data': data})

def scrape_usa_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'
    }
    url = "https://www.transfermarkt.us/major-league-soccer/besucherzahlen/wettbewerb/MLS1"
    
    response = requests.get(url, headers=headers)
    page_soup = BeautifulSoup(response.content, 'html.parser')
    
    teams_html = page_soup.find_all('tr', class_='odd') + page_soup.find_all('tr', class_='even')
    
    teams_data = []
    for team_html in teams_html:
        cells = team_html.find_all('td')
        
        if len(cells) >= 6:  
            teamname = cells[4].text.strip()  
            stadium = cells[3].text.strip()
            # Convert capacity and number of spectators to integers
            capacity = int(float(cells[5].text.strip().replace(",", "").replace(".", "")))
            numofspecators = int(float(cells[6].text.strip().replace(",", "").replace(".", "")))
            
            # Convert average to float
            average = int(float(cells[7].text.strip().replace(",", "").replace(".", ""))) 
            
            teams_data.append({
                'Team_Name': teamname,
                'Stadium': stadium,
                'Capacity': capacity,
                'Number_of_Spectators': numofspecators,
                'Average': average
            })
    return teams_data

@login_required(login_url='/authentication/login')
def usa_view(request):
    data = scrape_usa_data()
    #print(data)
    return render(request, 'webapp/usa.html', {'teams_data': data})

def scrape_mexico_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'
    }
    url = "https://www.transfermarkt.es/liga-mx-clausura/besucherzahlen/wettbewerb/MEX1"
    
    response = requests.get(url, headers=headers)
    page_soup = BeautifulSoup(response.content, 'html.parser')
    
    teams_html = page_soup.find_all('tr', class_='odd') + page_soup.find_all('tr', class_='even')
    
    teams_data = []
    for team_html in teams_html:
        cells = team_html.find_all('td')
        
        if len(cells) >= 6:  
            teamname = cells[4].text.strip()  
            stadium = cells[3].text.strip()
            # Convert capacity and number of spectators to integers
            capacity = int(float(cells[5].text.strip().replace(",", "").replace(".", "")))
            numofspecators = int(float(cells[6].text.strip().replace(",", "").replace(".", "")))
            
            # Convert average to float
            average = int(float(cells[7].text.strip().replace(",", "").replace(".", "")))            
            teams_data.append({
                'Team_Name': teamname,
                'Stadium': stadium,
                'Capacity': capacity,
                'Number_of_Spectators': numofspecators,
                'Average': average
            })
    return teams_data

@login_required(login_url='/authentication/login')
def mexico_view(request):
    data = scrape_mexico_data()
    #print(data)
    return render(request, 'webapp/mexico.html', {'teams_data': data})

@login_required(login_url='/authentication/login')
def profile(request):
    favorite_teams = UserFavorite.objects.filter(user=request.user).select_related('team')
    return render(request, 'webapp/profile.html', {'favorite_teams': favorite_teams})

@login_required(login_url='/authentication/login')
def knn_view(request):
    print("Rendering KNN view")
    return render(request, 'webapp/knn.html')

@login_required(login_url='/authentication/login')
def decision_tree_view(request):
    return render(request, 'webapp/tree.html')

@login_required(login_url='/authentication/login')
def linear_regression_view(request):
    return render(request, 'webapp/regression.html')


@login_required(login_url='/authentication/login')
def svm_view(request):
    return render(request, 'webapp/svm.html')

@login_required(login_url='/authentication/login')
def summary_view(request):
    return render(request, 'webapp/summary.html')