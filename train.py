import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

data = pd.read_csv('Forest_fire.csv')

print("First few rows of the dataset:")
print(data.head())

print("Columns with missing or invalid data:")
print(data[['Area', 'Oxygen', 'Temperature', 'Humidity']].isnull().sum())

data[['Area', 'Oxygen', 'Temperature', 'Humidity']] = data[['Area', 'Oxygen', 'Temperature', 'Humidity']].apply(pd.to_numeric, errors='coerce')

print("Columns with missing data after coercion:")
print(data[['Area', 'Oxygen', 'Temperature', 'Humidity']].isnull().sum())

data.fillna(data.mean(), inplace=True)

print(f"Dataset shape after cleaning: {data.shape}")

if data.empty:
    raise ValueError("The dataset is empty after cleaning. Please check the input data for validity.")

X = data[['Area', 'Oxygen', 'Temperature', 'Humidity']]
y = data['Fire Occurrence']

if not pd.api.types.is_numeric_dtype(y):
    raise ValueError("The target column 'Fire Occurrence' contains non-numeric values. Please clean the data.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

os.makedirs('models', exist_ok=True)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
