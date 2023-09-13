from django import forms


region = [('', ''), ('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]
season = [('', ''), ('Winter', 'Winter (LEC Only) '), ('Spring', 'Spring'), ('Summer', 'Summer')]

class ParameterForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(ParameterForm, self).__init__(*args, **kwargs)
        self.fields['min_games'].initial = 3 # doesn't work

    region_choice = forms.ChoiceField(widget=forms.Select, choices=region, initial='LJL', label='region')
    season_choice = forms.ChoiceField(widget=forms.Select, choices=season, label='season')
    # initial doesn't work
    min_games = forms.IntegerField(widget=forms.NumberInput, max_value=9, min_value=3, initial=3, label='min games')
    
    def clean(self):
        cleaned_data = super().clean()
        region = cleaned_data.get('region_choice')
        season = cleaned_data.get('season_choice')
        # min_games = cleaned_data.get('min_games')
        
        if (region != 'LEC') and (season == 'Winter'):
            raise forms.ValidationError('Winter Season はLECのみ有効です。')