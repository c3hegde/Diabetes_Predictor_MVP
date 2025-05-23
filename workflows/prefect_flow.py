#designed for monitoring in the Prefect UI. This version includes:  Task logging,  Retry logic,  Tracked steps,  Modular task breakdown,  Ready to run in Prefect Cloud


from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


# Configurable paths
DATA_PATH = 'data/raw/diabetes.csv'
MODEL_PATH = 'models/logistic_model.pkl'
SCALER_PATH = 'models/scaler.pkl'


@task(name="Load Dataset", retries=2, retry_delay_seconds=5, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_data():
    logger = get_run_logger()
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df
    pass


@task(name="Preprocess Data")
def preprocess_data(df):
    logger = get_run_logger()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler
    pass

@task(name="Train Logistic Regression Model")
def train_model(X_train, y_train):
    logger = get_run_logger()
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
    pass

@task(name="Save Model and Scaler")
def save_artifacts(model, scaler):
    logger = get_run_logger()
    os.makedirs("models", exist_ok=True)
    pickle.dump(model, open(MODEL_PATH, 'wb'))
    pickle.dump(scaler, open(SCALER_PATH, 'wb'))
    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"Scaler saved to {SCALER_PATH}")


@flow(name="diabetes-training-flow")
def diabetes_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = train_model(X_train, y_train)
    save_artifacts(model, scaler)
    print("✅ Flow completed.")


if __name__ == "__main__":
    diabetes_pipeline()
