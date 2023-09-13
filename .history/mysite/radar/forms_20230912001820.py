from django import forms


region = [('', ''), ('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]
season = [('', ''), ('Winter', 'Winter (LEC Only)'), ('Spring', 'Spring'), ('Summer', 'Summer')]
position = [('', ''), ('top', 'top'), ('jungle', 'jungle'), ('mid', 'mid'), ('bot', 'bot'), ('support', 'support')]

class ParameterForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super(ParameterForm, self).__init__(*args, **kwargs)
        self.fields['min_games'].initial = 3 # doesn't work

    region_choice = forms.ChoiceField(widget=forms.Select, choices=region, initial='', label='region')
    season_choice = forms.ChoiceField(widget=forms.Select, choices=season, initial='', label='season')
    position_choice = forms.ChoiceField(widget=forms.Select, choices=position, initial='', label='position')
    # initial doesn't work
    min_games = forms.IntegerField(widget=forms.NumberInput, max_value=9, min_value=3, initial=3, label='min games count')
    
    def clean(self):
        cleaned_data = super().clean()
        region_posted = cleaned_data.get('region_choice')
        season_posted = cleaned_data.get('season_choice')

        if (region_posted != 'LEC') and (season_posted == 'Winter'):
            raise forms.ValidationError(('! Winter Season is valid only at LEC'), code='invalid')