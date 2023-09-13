from django import forms


region = [('', ''), ('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]
season = [('', ''), ('Winter', 'Winter (LEC Only) '), ('Spring', 'Spring'), ('Summer', 'Summer')]

class ParameterForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(ParameterForm, self).__init__(*args, **kwargs)
        self.fields['min_games'].initial = 3 # doesn't work

    region_choice = forms.ChoiceField(widget=forms.Select, choices=region, initial='', label='region')
    season_choice = forms.ChoiceField(widget=forms.Select, choices=season, initial='', label='season')
    # initial doesn't work
    min_games = forms.IntegerField(widget=forms.NumberInput, max_value=9, min_value=3, initial=3, label='min games')
    
    def clean_winter(self):
        cleaned_data = super().clean()
        a = cleaned_data.get('region_choice')
        b = cleaned_data.get('season_choice')
        # min_games = cleaned_data.get('min_games')
        
        if (a != 'LEC') and (b == 'Winter'):
            raise forms.ValidationError(("Invalid value"), code="invalid")