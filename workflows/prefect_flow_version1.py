#This script will orchestrate the model training pipeline, automating steps like loading data, preprocessing, training, and saving the models.
#this is the first working version I used

from prefect import flow, task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Get path to the current file (prefect_flow.py)
base_path = os.path.dirname(os.path.abspath(__file__))
print (base_path)
# Full path to the data file
DATA_PATH = os.path.join(base_path, '..','data', 'raw', 'diabetes.csv')
MODEL_PATH = os.path.join(base_path, '..','models', 'logistic_model.pkl')
SCALER_PATH = os.path.join(base_path,'..', 'models', 'scaler.pkl')
# Use the full path to read the file
#with open(data_path, 'r') as f:

#DATA_PATH = "data/raw/diabetes.csv"
#MODEL_PATH = "models/logistic_model.pkl"
#SCALER_PATH = "models/scaler.pkl"

@task
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

@task
def preprocess_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

@task
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

@task
def save_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(scaler, open(scaler_path, 'wb'))

@flow(name="Diabetes Model Training Pipeline")
def training_pipeline():
    df = load_data()
    X_scaled, y, scaler = preprocess_data(df)
    model, accuracy = train_model(X_scaled, y)
    save_model(model, scaler)
    print(f"✅ Training complete. Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    training_pipeline()
