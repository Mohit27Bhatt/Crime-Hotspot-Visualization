from  django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class NewUserForm(UserCreationForm):
    email = forms.EmailField(required=True)
    age = forms.IntegerField(required=True)
    
    class Meta:
        
        model = User
        fields = ('username','email','password1','password2', 'age')
        
    def save(self, commit=True):
        user =  super(NewUserForm,self).save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user
    

class ApprovalStatusForm(forms.Form):
    APPROVAL_CHOICES = [
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]
    approval_status = forms.ChoiceField(choices=APPROVAL_CHOICES)

