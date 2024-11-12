from django import forms
from .models import Rating


class AddRatingForm(forms.ModelForm):
    
  
    class Meta:
        model = Rating
        fields = ['rating']
        labels = {'rating': 'Rating'}
        widgets={
            'rating':forms.TextInput(attrs={'type':'range','step':'0.5','min':'0.5','max':'5','class':{'custom-range','border-0'}})
        }


    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       self.fields['rating'].widget = forms.TextInput(attrs={'type': 'number', 'class': 'starability-basic', 'step':'0.5','min': '1', 'max': '5'})
