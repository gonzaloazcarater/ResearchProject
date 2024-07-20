import csv
from django.core.management.base import BaseCommand
from webapp.models import TeamData

class Command(BaseCommand):
    help = 'Import teams from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        self.import_teams_from_csv(csv_file)

    def import_teams_from_csv(self, file_path):
        # Delete all files prior import
        TeamData.objects.all().delete()
        with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                team = row.get('\ufeffTeam', row.get('Team', None))  # Access colums wihtout BOM
                if team is not None:
                    # Replace commas with poins
                    season_ticket_price = row.get('Season Ticket Avg Price(€)', '').replace('.', '').replace(',', '.')
                    avg_annual_salary = row.get('Avg Annual Salary (USD)', '').replace('.', '').replace(',', '.')
                    population_millions = row.get('Popuation(millions)', '').replace('.', '').replace(',', '.')
                    avg_attendance = row.get('Avg Attendance', '').replace('.', '').replace(',', '.')
                    capacity = row.get('Capacity', '')
                    number_of_spectators = row.get('Number of Spectators', '')

                    TeamData.objects.create(
                        Team=team,
                        Country=row.get('Country', ''),
                        Season_Ticket_Avg_Price_EUR=float(season_ticket_price) if season_ticket_price else 0.0,
                        Avg_Attendance=float(avg_attendance) if avg_attendance else 0.0,
                        Avg_Annual_Salary_USD=int(avg_annual_salary) if avg_annual_salary else 0,
                        Population_Millions=float(population_millions) if population_millions else 0.0,
                        Capacity=int(capacity) if capacity else 0,
                        Number_of_Spectators=int(number_of_spectators) if number_of_spectators else 0
                    )
        self.stdout.write(self.style.SUCCESS('Data imported successfully'))
