from django import forms


region = [('', '選択してください'), ('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]
season = [('', '選択してください'), ('Winter', 'Winter (LEC Only) '), ('Spring', 'Spring'), ('Summer', 'Summer')]

class ParameterForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(ParameterForm, self).__init__(*args, **kwargs)
        self.fields['min_games'].initial = 3 # doesn't work

    region_choice = forms.ChoiceField(widget=forms.Select, choices=region, initial='LJL', label='region')
    season_choice = forms.ChoiceField(widget=forms.Select, choices=season, label='season')
    # initial doesn't work
    min_games = forms.IntegerField(widget=forms.NumberInput, max_value=9, min_value=3, initial=3, label='min games')