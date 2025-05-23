# Diabetes Risk Predictor - AI MVP
A lightweight machine learning web app that predicts the likelihood of diabetes in patients using health metrics. Built with `scikit-learn`, `Streamlit`, and `Prefect`.

## Features

- Logistic Regression for prediction
- Interactive UI (Streamlit)
- Automated ML workflows with Prefect
- EDA & Clustering Analysis (KMeans)
- Deployed via Google Colab / Streamlit Cloud
- Lightweight & modular repo structure

## Use Case

This MVP demonstrates how healthcare providers or clinical staff can quickly screen patients for potential diabetes risk using minimal inputs.

## Models Used

| Algorithm           | Purpose            |
|---------------------|--------------------|
| Logistic Regression | Classification     |
| KMeans              | Patient Segmentation |
| StandardScaler      | Feature Scaling    |

## Project Structure

diabetes-predictor-ai-mvp/
├── data/               # Raw and processed datasets
├── notebooks/          # EDA and model training
├── app/                # Streamlit app and utilities
├── models/             # Trained models (pkl)
├── workflows/          # Prefect orchestration

Demo Video: Click here to watch the demo

##Future Enhancements
    Add drift detection and retraining triggers
    Host on AWS with Lambda or EC2
    Incorporate medical notes using LLM + RAG