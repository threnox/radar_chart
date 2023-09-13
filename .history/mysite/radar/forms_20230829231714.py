from django import forms


region = [('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]
season = [('Winter', 'Winter(LEC Only)'), ('Spring', 'Spring'), ('Summer', 'Summer')]


class ParameterForm(forms.Form):
    # text = forms.CharField(label='地域')
    region_choice = forms.ChoiceField(widget=forms.Select, choices=region, initial='LJL', label='地域')
    season_choice = forms.ChoiceField(widget=forms.Select, choices=season, initial='Spring', label='シーズン')
    min_games = forms.IntegerField(widget=forms.NumberInput, initial=3, max_value=9, min_value=1, label='最少出場ゲーム数')