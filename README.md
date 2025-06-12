# Diabetes Risk Predictor - MVP
A lightweight machine learning web app that predicts the likelihood of diabetes in patients using health metrics. Built with `scikit-learn`, `Streamlit`, and `Prefect`.

## Features

- Logistic Regression for prediction
- Interactive UI (Streamlit)
- Automated ML workflows with Prefect
- EDA & Clustering Analysis (KMeans)
- Deployed via Google Colab / Streamlit Cloud
- Dataset taken from PIMA Diabetes Dataset
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
- ├── data/               # Raw and processed datasets
- ├── notebooks/          # Exploratory Data Analysis and model training
- ├── app/                # Streamlit app and utilities
- ├── models/             # Trained models (pkl)
- ├── workflows/          # Prefect orchestration

## Potential Future Enhancements
- Host on AWS with Lambda or EC2
- Incorporate medical notes using LLM + RAG
- Add drift detection and retraining triggers

