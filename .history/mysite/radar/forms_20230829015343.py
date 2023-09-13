from django import forms


region_choice = [('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]

class RegionForm(forms.Form):
    text = forms.CharField(label='地域')
    region = forms.ChoiceField(widget=forms.Select, choices=region_choice, initial='LJL')