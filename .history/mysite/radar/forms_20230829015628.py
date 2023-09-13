from django import forms


region = [('LCK', 'LCK'), ('LEC', 'LEC'), ('LJL', 'LJL')]

class RegionForm(forms.Form):
    text = forms.CharField(label='地域')
    region_choice = forms.ChoiceField(widget=forms.Select, choices=region, initial='LJL', label='地域')