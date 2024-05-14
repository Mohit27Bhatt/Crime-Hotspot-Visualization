from django.shortcuts import render,redirect
from .forms import NewUserForm
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect

#from .models import User


# Create your views here.
def homepage(request):
    return render(request,'homepage.html')

def crime_in_india(request):
    return render(request,'homepage.html')

def data_and_visualisation(request):
    return render(request,'data_and_visualisation.html')

def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')


def register(request):
    
    if request.method == 'POST':
        print("k")
        form = NewUserForm(request.POST)
        if form.is_valid():
            print("kkk")
            form.save()
            messages.success(request, 'Registration Successful. Please log in.')
            return redirect("login")  # Redirect to the login page after successful registration
        else:
            print("no re")
            messages.error(request, 'Registration unsuccessful. Please correct the errors below.')
    else:
        form = NewUserForm()
        print("gg")
    return render(request=request, template_name='register.html', context={'register_form': form})



# login page

from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect

def custom_login(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                auth_login(request, user)  # Use Django's login function
                messages.info(request, f"You are now logged in as {username}.")
                return redirect("admin_dashboard")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:  # Handle GET request
        form = AuthenticationForm()
        return render(request=request, template_name='login.html', context={"login_form": form})


from django.contrib.auth import logout as auth_logout
from django.shortcuts import redirect

def custom_logout(request):
    auth_logout(request)
    # Redirect to a specific URL after logout
    return redirect('homepage')  # Assuming 'home' is the name of your home page URL pattern


def userhome(request):
    return render(request,'userhome.html')

def view(request):
    global df
    df = pd.read_excel('crimeapp/20230320020226crime_data_extended_entries.xlsx')
    col = df.head(1000).to_html
    return render(request, "view.html", {'table': col})





def moduless(request):
    global df,x_train, x_test, y_train, y_test 
    if request.method == "POST":
        model = request.POST['algo']

        if model == "2":
            df = pd.read_excel('crimeapp/20230320020226crime_data_extended_entries.xlsx')
            #Delete a unknown column
            df.drop("date",axis=1,inplace=True)
            df.drop("time_of_day",axis=1,inplace=True)
            df.drop("latitude",axis=1,inplace=True)
            df.drop("longitude",axis=1,inplace=True)
            le = LabelEncoder()
            col = df[['crime_type','location','victim_gender','perpetrator_gender','weapon','injury','weather','temperature','previous_activity']]
            for i in col:
                df[i]=le.fit_transform(df[i])
            x = df.drop(['crime_type'], axis = 1) 
            y = df['crime_type']
            Oversample = RandomOverSampler(random_state=72)
            x_sm, y_sm = Oversample.fit_resample(x[:100],y[:100])
            x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size = 0.4, random_state= 42)
            re = RandomForestClassifier(ccp_alpha=0.035)
            re.fit(x_train,y_train)
            re_pred = re.predict(x_train)
            ac = accuracy_score(y_train,re_pred)
            ac
            msg='Accuracy of RandomForest : ' + str(ac)
            return render(request,'moduless.html',{'msg':msg})
        elif model == "1":
            df = pd.read_excel('crimeapp/20230320020226crime_data_extended_entries.xlsx')
            #Delete a unknown column
            df.drop("date",axis=1,inplace=True)
            df.drop("time_of_day",axis=1,inplace=True)
            df.drop("latitude",axis=1,inplace=True)
            df.drop("longitude",axis=1,inplace=True)
            le = LabelEncoder()
            col = df[['crime_type','location','victim_gender','perpetrator_gender','weapon','injury','weather','temperature','previous_activity']]
            for i in col:
                df[i]=le.fit_transform(df[i])
            x = df.drop(['crime_type'], axis = 1) 
            y = df['crime_type']
            Oversample = RandomOverSampler(random_state=72)
            x_sm, y_sm = Oversample.fit_resample(x[:100],y[:100])
            x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size = 0.35, random_state= 42)
            de = DecisionTreeClassifier(ccp_alpha=0.015)
            de.fit(x_train,y_train)
            de_pred = de.predict(x_train)
            ac1 = accuracy_score(y_train,de_pred)
            ac1
            msg='Accuracy of Decision tree : ' + str(ac1)
            return render(request,'moduless.html',{'msg':msg})
        elif model == "3":
            df = pd.read_excel('crimeapp/20230320020226crime_data_extended_entries.xlsx')
            df.drop("longitude",axis=1,inplace=True)
            df.drop("latitude",axis=1,inplace=True)
            le = LabelEncoder()
            col = df[['date','time_of_day','crime_type','location','victim_gender','perpetrator_gender','weapon','injury','weather','temperature','previous_activity']]
            for i in col:
                df[i]=le.fit_transform(df[i])
            x = df.drop(['crime_type'], axis = 1) 
            y = df['crime_type']
            Oversample = RandomOverSampler(random_state=72)
            x_sm, y_sm = Oversample.fit_resample(x[:100],y[:100])

            x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size = 0.2, random_state= 42)
            gd = GradientBoostingClassifier()
            gd.fit(x_train,y_train)
            gd_pred = gd.predict(x_test)
            bc = accuracy_score(y_test,gd_pred)
            bc
            msg='Accuracy of GradientBoostingClassifier : ' + str(bc)
            return render(request,'moduless.html',{'msg':msg})
    return render(request,'moduless.html')


def prediction(request):
    try:
        global df,x_train, x_test, y_train, y_test
        print("Request mdethod:", request.method)
        
        a = float(request.POST['f1'])
        # b = float(request.POST['f2'])
        # c = float(request.POST['f3'])
        d = float(request.POST['f4'])
        e = float(request.POST['f5'])
        f = float(request.POST['f6'])
        g = float(request.POST['f7'])
        h = float(request.POST['f8'])
        i = float(request.POST['f9'])
        j = float(request.POST['f10'])
        k = float(request.POST['f11'])
        l = float(request.POST['f12'])
        print("fv1")
        l1 = [[a,d,e,f,g,h,i,j,k,l]]
        de = DecisionTreeClassifier()
        print("fv2")
        print(l1)
        de.fit(x_train,y_train)
        print("d")
        pred = de.predict(l1)
        print(pred)
        if pred == 0:
            msg = 'Robbery'
        elif pred == 1:
            msg = 'Embezzlement'
        elif pred == 2:
            msg = 'Burglary'
        elif pred == 3:
            msg = 'Vandalism'
        elif pred == 4:
            msg = 'Theft'
        elif pred == 5:
            msg = 'Assault'
        elif pred == 6:
            print('Forgery')
        elif pred == 7:
            msg ='Drug Offense'
        else:
            msg = 'Fraud'
        print("kk")
        if a == 1:
            lat = 12.9255
            lag = 77.5468
            name = "Banashankari"
        elif a == 2:
            lat = 12.9304
            lag = 77.6784
            name = "Bellandur"
        elif a == 3:
            lat = 12.8452 
            lag = 77.6602
            name = "Electronic City"
        elif a == 4:
            lat = 12.9121  
            lag = 77.6446
            name = "HSR layout"
        elif a == 5:
            lat = 12.9784
            lag = 77.6408
            name = "Indiranagar"
        elif a == 6:
            lat =  12.9308
            lag =  77.5838
            name = "jayanagar"
        elif a == 7:
            lat = 12.9063
            lag = 77.5857
            name = "jp nagar"
        elif a == 8:
            lat = 12.9855
            lag = 77.5269
            name = "Kamakshipalya"
        elif a == 9:
            lat = 12.9352
            lag = 77.6245
            name = "Koramangala"
        elif a == 10:
            lat = 12.9569
            lag = 77.7011
            name = "Marathahalli"
        elif a == 11:
            lat = 12.9698
            lag = 77.7500
            name = "White Field"
        elif a == 12:
            lat = 13.1155
            lag = 77.6070
            name = "White Field"

        print(lat)
        print(lag)
        import folium
        m = folium.Map(location=[19,-12],zoom_start=2)
        folium.Marker([lat,lag],tooltip='click for more',popup=name).add_to(m)
        m = m._repr_html_()
        print(msg)
        return render(request,'result.html',{'msg':msg,'m':m})
    except:
        msg = "Please give a required input"
        return render(request,'prediction.html',{'msg':msg})
        

    return render(request,'prediction.html')



from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def know_your_state(request):
    return render(request,'know_your_state.html')






# views.py
import pandas as pd
from django.shortcuts import render
from django.views import View
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

class andhra_pradesh(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Andhra Pradesh']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Train Support Vector Machine (SVM) model for stacking
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model for stacking
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a stacking regressor with SVM and XGBoost
            stack_model = StackingRegressor(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model)
                ],
                final_estimator=LinearRegression()
            )
            
            # Fit the stacking regressor
            stack_model.fit(X, y)
            
            # Predict using the stacking regressor
            predicted_crime_stack = stack_model.predict([[2023]])
            accuracy_stack = stack_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'stacking': {'prediction': predicted_crime_stack[0], 'accuracy': accuracy_stack},
                    'svm + XGB': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'andhra_pradesh.html', context)





from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

class arunachal_pradesh(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Arunachal Pradesh
        arunachal_data = df[df['State/UT'] == 'Arunachal Pradesh']
        
        # Group data by crime category
        grouped_data = arunachal_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Train MLP Regressor model
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'arunachal_pradesh.html', context)

    


class assam(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Assam']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'assam.html', context)





class bihar(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Bihar']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'bihar.html', context)
    


    
class chhattisgarh(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Chhattisgarh']
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'chhattisgarh.html', context)



class goa(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Goa']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'goa.html', context)
    

class gujarat(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Gujarat']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'gujarat.html', context)
    

class haryana(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Haryana']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'haryana.html', context)



class himachal_pradesh(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Himachal Pradesh']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'himachal_pradesh.html', context)



class jharkhand(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Jharkhand']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'jharkhand.html', context)



class karnataka(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Karnataka']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'karnataka.html', context)



class kerala(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Kerala']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'kerala.html', context)



class madhya_pradesh(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Madhya Pradesh']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'madhya_pradesh.html', context)



class maharashtra(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Maharashtra']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'maharashtra.html', context)



class manipur(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Manipur']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'manipur.html', context)



class meghalaya(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Meghalaya']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'meghalaya.html', context)



class mizoram(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Mizoram']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'mizoram.html', context)



class nagaland(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Nagaland']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'nagaland.html', context)



class odisha(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Odisha']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'odisha.html', context)



class punjab(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Punjab']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'punjab.html', context)



class rajasthan(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Rajasthan']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'rajasthan.html', context)



class sikkim(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Sikkim']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'sikkim.html', context)



class tamil_nadu(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Tamil Nadu']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'tamil_nadu.html', context)



class telangana(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Telangana']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'telangana.html', context)



class tripura(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Tripura']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'tripura.html', context)



class uttar_pradesh(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Uttar Pradesh']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'uttar_pradesh.html', context)



class uttarakhand(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Uttarakhand']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'uttarakhand.html', context)



class west_bengal(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'West Bengal']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'west_bengal.html', context)



class an_islands(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'A&N Islands']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'a&n_islands.html', context)



class chandigarh(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Chandigarh']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'chandigarh.html', context)



class dadra_and_nagar_haveli_and_daman_and_diu(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'D&N Haveli and Daman & Diu']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'dadra_and_nagar_haveli_and_daman_and_diu.html', context)



class delhi(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Delhi']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'delhi.html', context)



class lakshadweep(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Lakshadweep']
        
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'lakshadweep.html', context)



class puducherry(View):
    def get(self, request):
        # Read data from Excel file
        df = pd.read_excel('crimeapp/crime_data.xlsx')
        
        # Filter data for Andhra Pradesh
        andhra_data = df[df['State/UT'] == 'Puducherry']
        # Group data by crime category
        grouped_data = andhra_data.groupby('category')
        
        # List to store predictions and accuracies for each category
        predictions = []
        # Iterate over each crime category
        for category, data in grouped_data:
            # Extract features (years) and target (crime data)
            years = data.columns[1:-4].astype(int)
            crime_data = data.iloc[0, 1:-4].astype(int)
            
            # Reshape the data for model training
            X = years.values.reshape(-1, 1)
            y = crime_data.values
            
            # Train Linear Regression model
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            predicted_crime_linear = linear_model.predict([[2023]])
            accuracy_linear = r2_score(y, linear_model.predict(X))
            
            # Train Ridge Regression model
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X, y)
            predicted_crime_ridge = ridge_model.predict([[2023]])
            accuracy_ridge = r2_score(y, ridge_model.predict(X))
            
            # Train Lasso Regression model
            lasso_model = Lasso(alpha=1.0)
            lasso_model.fit(X, y)
            predicted_crime_lasso = lasso_model.predict([[2023]])
            accuracy_lasso = r2_score(y, lasso_model.predict(X))
            
            # Train Random Forest Regression model
            rf_model = RandomForestRegressor(n_estimators=100)
            rf_model.fit(X, y)
            predicted_crime_rf = rf_model.predict([[2023]])
            accuracy_rf = r2_score(y, rf_model.predict(X))
            
            # Store predictions and accuracies for this category
            svm_model = SVR(kernel='rbf')
            svm_model.fit(X, y)
            predicted_crime_svm = svm_model.predict([[2023]])
            accuracy_svm = svm_model.score(X, y)
            
            # Train Extreme Gradient Boosting (XGBoost) model
            xgb_model = XGBRegressor()
            xgb_model.fit(X, y)
            predicted_crime_xgb = xgb_model.predict([[2023]])
            accuracy_xgb = xgb_model.score(X, y)
            
            # Create a voting regressor with Random Forest, SVM, and XGBoost
            voting_regressor = VotingRegressor(
                estimators=[
                    ('rf', rf_model),
                    #('svm', svm_model),
                    ('xgb', xgb_model)
                ]
            )
            
            # Fit the voting regressor
            voting_regressor.fit(X, y)
            
            # Predict using the voting regressor
            predicted_crime_voting = voting_regressor.predict([[2023]])
            accuracy_voting = voting_regressor.score(X, y)
            
            # Store predictions and accuracies for this category
            predictions.append({
                'category': category,
                'models': {
                    'linear': {'prediction': predicted_crime_linear[0], 'accuracy': accuracy_linear},
                    'lasso': {'prediction': predicted_crime_lasso[0], 'accuracy': accuracy_lasso},
                    'ridge': {'prediction': predicted_crime_ridge[0], 'accuracy': accuracy_ridge},
                    'rf': {'prediction': predicted_crime_rf[0], 'accuracy': accuracy_rf},
                    #'svm': {'prediction': predicted_crime_svm[0], 'accuracy': accuracy_svm},
                    #'xgb': {'prediction': predicted_crime_xgb[0], 'accuracy': accuracy_xgb},
                    'rf+ xgb': {'prediction': predicted_crime_voting[0], 'accuracy': accuracy_voting},
                }
            })
        
        # Pass data to template
        context = {
            'predictions': predictions,
        }
        
        return render(request, 'puducherry.html', context)

import json
import pandas as pd
import folium
from django.shortcuts import render

def crime_hotspots(request):
    excel_file_path = 'crimeapp/crime_data.xlsx'

    # Read Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)

    # Define the crime categories
    crime_categories = [
        'murder',
        'Kidnapping and Abduction',
        'Crime Against Women',
        'Crime Against Children',
        'Juveniles in Conflict with Law',
        'Crime Against Senior Citizen',
        'Crime Against Scheduled Castes',
        'Crime Against Scheduled Tribes',
        'Economic Offences',
        'Corruption Offences',
        'Cyber Crimes'
    ]

    # Create a dictionary to store maps for each crime category
    maps = {}

    # Process data for each crime category
    for crime_category in crime_categories:
        # Filter DataFrame for the current crime category
        category_df = df[df['category'] == crime_category]

        # Group DataFrame by 'State/UT' and sum 'Total_crime' for each state
        state_totals = category_df.groupby('State/UT')['Total_crime'].sum().reset_index()

        # Create a Folium map
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=7)

        # Choropleth map layer with custom color scale
        folium.Choropleth(
            geo_data='crimeapp/india_state_geo.json',
            data=state_totals,
            columns=['State/UT', 'Total_crime'],
            key_on='feature.properties.NAME_1',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Total Crimes'
        ).add_to(m)

        # Convert the map to HTML
        map_html = m._repr_html_()

        # Store the map HTML in the dictionary
        maps[crime_category] = map_html

    return render(request, 'crime_hotspot.html', {'maps': maps})

import csv
from django.shortcuts import render
from django.http import HttpResponse

def report_crime(request):
    return render(request, 'report_crime.html')

def save_crime_report(request):
    if request.method == 'POST':
        crime_type = request.POST.get('crime_type')
        date = request.POST.get('date')
        time = request.POST.get('time')
        inputState = request.POST.get('selected_state')
        inputDistrict = request.POST.get('selected_district')
        latitude=request.POST.get('latitude')
        longitude = request.POST.get('longitude')

        # Write data to CSV file
        with open('approved_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([crime_type, date, time, inputState, inputDistrict, latitude, longitude ,'pending'])

        return HttpResponse('Crime report saved successfully.')

    return HttpResponse('Method not allowed.')

import csv
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from datetime import datetime

def sort_data_by_time(data):
    # Sort the data by the time of the date
    sorted_data = sorted(data[1:], key=lambda x: datetime.strptime(x[1] + ' ' + x[2], '%Y-%m-%d %H:%M'))
    return [data[0]] + sorted_data

from reportlab.platypus import Spacer

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer
from django.http import HttpResponse
import csv

from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

def calculate_most_affected_state(data):
    state_counts = {}
    for row in data[1:]:
        state = row[3]
        if state in state_counts:
            state_counts[state] += 1
        else:
            state_counts[state] = 1

    print(state_counts)
    most_affected_state = max(state_counts, key=state_counts.get)
    percentage = (state_counts[most_affected_state] / len(data[1:])) * 100
    return f"The most affected state is {most_affected_state} with {percentage:.2f}% of crimes."


def find_hotspot_coordinates(data):
    latitude_counts = {}
    longitude_counts = {}
    for row in data[1:]:
        latitude = float(row[5])  # Assuming latitude is at index 5
        longitude = float(row[6])  # Assuming longitude is at index 6
        for i in range(-1, 2):  # Check within range of ±1 for both latitude and longitude
            if latitude + i not in latitude_counts:
                latitude_counts[latitude + i] = 0
            if longitude + i not in longitude_counts:
                longitude_counts[longitude + i] = 0
            latitude_counts[latitude + i] += 1
            longitude_counts[longitude + i] += 1
    
    most_frequent_latitude = max(latitude_counts, key=latitude_counts.get)
    most_frequent_longitude = max(longitude_counts, key=longitude_counts.get)
    
    return f"The hotspot of crime is at latitude {most_frequent_latitude} and longitude {most_frequent_longitude}."




def find_most_frequent_crime_type(data):
    crime_type_counts = {}
    for row in data[1:]:
        crime_type = row[0]
        if crime_type in crime_type_counts:
            crime_type_counts[crime_type] += 1
        else:
            crime_type_counts[crime_type] = 1
    
    most_frequent_crime_type = max(crime_type_counts, key=crime_type_counts.get)
    return f"The most frequent crime type is {most_frequent_crime_type}."

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph

def spacet():
    return f"   \n"
from datetime import datetime, timedelta
def filter_last_7_days(data):
    #print(data)
    current_date = datetime.now()
    seven_days_ago = current_date - timedelta(days=7)
    filtered_data = []
     # Keep the header row
    for row in data[0:]:
        # Assuming date is in column 2 and time is in column 3
        row_date_time = datetime.strptime(f'{row[1]} {row[2]}', '%Y-%m-%d %H:%M')
        #print(row_date_time)
        #print(seven_days_ago)
        if row_date_time >= seven_days_ago:
            filtered_data.append(row)
        #print(filtered_data)
    return filtered_data

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph, PageBreak, Image
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

def download_report_pdf(request):
    # Set response content type as PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="crime_report.pdf"'

    # Create PDF document
    pdf = SimpleDocTemplate(response, pagesize=letter)

    # Title styling
    title_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                              ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                              ('FONTSIZE', (0, 0), (-1, -1), 20),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 12)])

    # Title data styling
    title_data_style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), (0.2, 0.6, 0.8)),
                                   ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),
                                   ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                   ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                                   ('FONTSIZE', (0, 0), (-1, -1), 20),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 25)])

    # Title data with style
    title_data = [['Crime Report']]

    # Apply title data style to title data
    title_table = Table(title_data)
    title_table.setStyle(title_data_style)

    # Data to populate the PDF
    data = []

    # Open the CSV file and read its data
    with open('approved_data.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        # Iterate over each row in the CSV file and append it to the data list
        for row in reader:
            row=row[:-1]
            row[5] = float(row[5])  # Latitude
            row[6] = float(row[6])
            data.append(row)

    # Sort the data by time
    data = filter_last_7_days(data)
    data = sort_data_by_time(data)

    # Rename table headers
    header_row = ['Crime Type', 'Date', 'Time', 'State', 'District', 'latitude', 'longitude']
    data.insert(0, header_row)

    # Table styling
    table_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                              ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                              ('BACKGROUND', (0, 0), (-1, 0), (0.8, 0.8, 0.8)),
                              ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),
                              ('GRID', (0, 0), (-1, -1), 1, (0, 0, 0))])

    # Table data styling
    table_data_style = TableStyle([('BACKGROUND', (0, 0), (-1, -1), (0.9, 0.9, 0.9)),
                                   ('TEXTCOLOR', (0, 0), (-1, -1), (0, 0, 0)),
                                   ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                   ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                                   ('FONTSIZE', (0, 0), (-1, -1), 9),
                                   ('LEFTPADDING', (0, 0), (-1, -1), 3),
                                   ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                                   ('TOPPADDING', (0, 0), (-1, -1), 6),
                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 6)])

    # Create table from the data
    table = Table(data)
    table.setStyle(table_style)
    table.setStyle(table_data_style)  # Apply data styling

    # Add space between title and table
    space = Spacer(1, 20)  # 20 units of space

    title_paragraph_style = ParagraphStyle(
        name='TitleParagraph',
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=(1, 1, 1),  # White color for text
        alignment=1,  # Center alignment
        spaceAfter=25  # Space after the paragraph
    )

    # Define content paragraph style
    content_paragraph_style = ParagraphStyle(
        name='ContentParagraph',
        fontName='Helvetica',
        fontSize=12,
        textColor=(0, 0, 0),  # Black color for text
        alignment=0,  # Left alignment
        spaceBefore=6,  # Top padding
        spaceAfter=6,   # Bottom padding
        leftIndent=10   # Left padding
    )

    # Additional sections
    sections = [
        ("1. Most Affected State", calculate_most_affected_state(data)),
        (" ", spacet()),
        ("2. Hotspot of Crime", find_hotspot_coordinates(data)),
        (" ", spacet()),
        ("3. Most Frequent Crime Type", find_most_frequent_crime_type(data))
    ]

    # Convert sections to PDF elements
    section_elements = [
        Paragraph(f'<para>{title}</para>', title_paragraph_style) for title, _ in sections
    ]
    section_elements += [
        Paragraph(content, content_paragraph_style) for _, content in sections
    ]

    crime_data = pd.read_csv('approved_data.csv')

    # Convert 'Date' column to datetime
    crime_data['Date'] = pd.to_datetime(crime_data['Date'])

    # Set 'Date' column as index
    crime_data.set_index('Date', inplace=True)

    # Aggregate data by day (counting the number of crimes per day)
    crime_daily = crime_data.resample('D').size()

    # Define and train the ARIMA model
    model = ARIMA(crime_daily, order=(49, 1, 0))
    model_fit = model.fit()

    # Forecast future crime rates
    forecast_steps = 7  # Forecast for the next 7 days
    forecast = model_fit.forecast(steps=forecast_steps)
    mae = mean_absolute_error(crime_daily[-forecast_steps:], forecast)
    mse = mean_squared_error(crime_daily[-forecast_steps:], forecast)
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')

    # Plot actual data for the previous 7 days and forecasted values for the next 7 days
    plt.figure(figsize=(13,9 ))
    plt.plot(crime_daily.index[-30:], crime_daily[-30:], label='Actual')  # Plot previous 7 days
    plt.plot(pd.date_range(start=crime_daily.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D'), forecast, color='red', label='Forecast')  # Plot forecast
    plt.title('Crime Rate Forecast')
    plt.xlabel('Date')
    plt.ylabel('Crime Count')
    plt.legend()
    plt.xticks(rotation=45)
    # Save the plot to a buffer
    plt.savefig('crime_forecast.png')

# Close the plot to free up memory
    plt.close()

    # Add the plot image to the PDF
    pdf_chart = Image('crime_forecast.png', width=400, height=300)  # Adjust width and height if needed
    elements = [title_table, space, table, pdf_chart] + section_elements  # Combine title table, space, data table, chart, and sections
    pdf.build(elements)
    return response

# views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
import csv
import os

@login_required
def admin_dashboard(request):
    # Read data from CSV file
    with open('crime_reports.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    if request.method == 'POST':
        approved_data = []
        rejected_data = []
        pending_data = []

        for index, record in enumerate(data):
            current_status = record.get('Approval Status')
            new_status = request.POST.get(f'approval_status_{index}')
            
            if new_status in ['pending', 'approved', 'rejected']:
                record['Approval Status'] = new_status

                # Separate records based on status
                if new_status == 'approved':
                    approved_data.append(record)
                elif new_status == 'rejected':
                    rejected_data.append(record)
                else:
                    pending_data.append(record)

        # Write to corresponding CSV files based on status
        if approved_data:
            with open('approved_data.csv', mode='a', newline='') as approved_file:
                writer = csv.DictWriter(approved_file, fieldnames=data[0].keys())
                if os.stat('approved_data.csv').st_size == 0:  # Write header only if file is empty
                    writer.writeheader()
                writer.writerows(approved_data)

        if rejected_data:
            with open('rejected_data.csv', mode='a', newline='') as rejected_file:
                writer = csv.DictWriter(rejected_file, fieldnames=data[0].keys())
                if os.stat('rejected_data.csv').st_size == 0:  # Write header only if file is empty
                    writer.writeheader()
                writer.writerows(rejected_data)

        # Rewrite the data back to the CSV file excluding approved and rejected records
        with open('crime_reports.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(pending_data)

    else:
        # Filter only pending status records
        pending_data = [record for record in data if record['Approval Status'] == 'pending']

    return render(request, 'admin_dashboard.html', {'data': pending_data})
