from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create your views here.
def index(request):
  return render(request,'index.html')

from django.http import HttpResponse
import numpy as np
import numpy as np  # Add this line for NumPy
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
path = "C:\\Users\\tharun\\Desktop\\folder\\hospital-prediction\\GUI\\training_data.csv"
dataset = pd.read_csv(path)

# Handle missing values in the dataset
dataset = dataset.dropna()

# Prepare input and output
inputs = dataset.drop('Survived_1_year', axis=1)
output = dataset['Survived_1_year']

# One-hot encode categorical variables
inputs_encoded_df = pd.get_dummies(inputs, columns=['Treated_with_drugs', 'Patient_Smoker', 'Patient_Rural_Urban', 'Patient_mental_condition'], drop_first=True)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(inputs_encoded_df, output, train_size=0.8)

# Train the model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

def training_data(request):
    if request.method == "POST":
        data = request.POST
        try:
            # Convert input values to appropriate types
            ID_Patient_Care_Situation = int(data.get('textID_Patient_Care_Situation'))
            Diagnosed_Condition = int(data.get('textDiagnosed_Condition'))
            # ... (similar conversion for other input variables)

            if 'buttonpredict' in request.POST:
                treated_drugs = data.get('textTreated_with_drugs').split()
                new_data = {
                    # ... (same as your existing code)
                }

                for drug in treated_drugs:
                    new_data[f'Treated_with_drugs_{drug}'] = [1]

                new_data_df = pd.DataFrame(new_data)

                # Ensure all columns are present in the new_data_df
                for column in inputs_encoded_df.columns:
                    if column not in new_data_df.columns:
                        new_data_df[column] = 0

                # Drop any additional columns not present in the model
                new_data_df = new_data_df[x_train.columns]

                # Check for NaN or infinity in the input data
                if new_data_df.isnull().values.any() or not np.isfinite(new_data_df.values).all():
                    raise ValueError("Input contains NaN, infinity, or a value too large for dtype('float32').")

                # Make prediction
                result = model.predict(new_data_df)

                # Render the result in the template
                return render(request, 'training_data.html', context={'result': "Survived_1_year=" + str(result)})

        except ValueError as e:
            return render(request, 'training_data.html', context={'error': str(e)})

    # Render the form
    return render(request, 'training_data.html')

 