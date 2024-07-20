from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Webapp(models.Model):
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField()

class TeamData(models.Model):
    Team = models.CharField(max_length=100)
    Country = models.CharField(max_length=100)
    Season_Ticket_Avg_Price_EUR = models.FloatField()
    Avg_Attendance = models.FloatField()
    Avg_Annual_Salary_USD = models.IntegerField()
    Population_Millions = models.FloatField()
    Capacity = models.IntegerField(default=0)
    Number_of_Spectators = models.IntegerField(default=0)
    def __str__(self):
        return self.Team
    
class UserFavorite(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    team = models.ForeignKey(TeamData, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'team')