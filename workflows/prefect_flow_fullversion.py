#This script will orchestrate the model training pipeline, automating steps like loading data, preprocessing, training, and saving the models.
#Fully automates data loading, preprocessing, training, saving
#How to Run This
    #Activate Prefect Cloud or local agent (optional): In command prompt
    #prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
#Run the script: In command prompt
#py prefect_flow_fullversion.py
#Optional Prefect UI & Scheduling Setup
#If you want to monitor or schedule this pipeline:
#prefect deployment build workflows/prefect_flow.py:training_pipeline -n retrain-weekly --cron "0 0 * * 0"
#prefect deployment apply training_pipeline-deployment.yaml
#prefect agent start



from prefect import flow, task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import pickle
import os

DATA_PATH = "data/raw/diabetes.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@task
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@task
def preprocess_data(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

@task
def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"✅ Logistic Regression Accuracy: {accuracy:.2f}")
    return model

@task
def train_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

@task
def save_model(obj, filename):
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(obj, f)

@flow(name="Diabetes Training Pipeline")
def training_pipeline():
    df = load_data()
    X_scaled, y, scaler = preprocess_data(df)
    
    model = train_logistic_regression(X_scaled, y)
    save_model(model, "logistic_model.pkl")
    save_model(scaler, "scaler.pkl")
    
    kmeans_model = train_kmeans(X_scaled)
    save_model(kmeans_model, "kmeans_model.pkl")

if __name__ == "__main__":
    training_pipeline()
