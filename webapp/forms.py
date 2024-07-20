from django import forms
from .models import TeamData, UserFavorite

class ToggleFavoriteForm(forms.Form):
    team_id = forms.IntegerField(widget=forms.HiddenInput())
    is_favorite = forms.BooleanField(required=False, widget=forms.HiddenInput())

    def toggle_favorite(self, user):
        team_id = self.cleaned_data.get('team_id')
        is_favorite = self.cleaned_data.get('is_favorite')

        try:
            team = TeamData.objects.get(id=team_id)
            user_favorite = UserFavorite.objects.filter(user=user, team=team).first()
            
            if is_favorite:
                if user_favorite:# If already a favorite, delete it
                    user_favorite.delete()
                    print(f"Team '{team.Team}' removed from favorites for user {user.username}")
                else:# If not a favorite, add it
                    UserFavorite.objects.create(user=user, team=team)
                    print(f"Team '{team.Team}' marked as favorite for user {user.username}")
            else:
                if user_favorite:# If already a favorite, delete it
                    user_favorite.delete()
                    print(f"Team '{team.Team}' removed from favorites for user {user.username}")
                else:# If not a favorite, add it
                    UserFavorite.objects.create(user=user, team=team)
                    print(f"Team '{team.Team}' marked as favorite for user {user.username}")
            return True
        except TeamData.DoesNotExist:
            return False
