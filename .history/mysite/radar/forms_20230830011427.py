from django import forms


region = [('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]
season = [('Winter', 'Winter(LEC Only)'), ('Spring', 'Spring'), ('Summer', 'Summer')]


class ParameterForm(forms.Form):
    region_choice = forms.ChoiceField(widget=forms.Select, choices=region, initial='LCK', label='region')
    season_choice = forms.ChoiceField(widget=forms.Select, choices=season, initial='Summer', label='season')
    min_games = forms.IntegerField(widget=forms.NumberInput, initial=4, max_value=9, min_value=1, label='min games')