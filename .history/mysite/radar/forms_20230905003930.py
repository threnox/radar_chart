from django import forms


region = [('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]
season = [('Winter', 'Winter (LEC Only) '), ('Spring', 'Spring'), ('Summer', 'Summer')]


class ParameterForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(ParameterForm, self).__init__(*args, **kwargs)
        self.fields['region_choice'].initial = 'LJL'
    #     self.fields['season_choice'].initial = 'Summer'
    #     self.fields['min_games'].initial = 4

    region_choice = forms.ChoiceField(widget=forms.Select, choices=region, initial='LJL', label='region', empty_value='選択してください')
    season_choice = forms.ChoiceField(widget=forms.Select, choices=season, label='season')
    min_games = forms.IntegerField(widget=forms.NumberInput, max_value=9, min_value=4, label='min games')