# Deployment

This folder contains the deployment files for the **Airline Recommendation Predictor**.

The application consists of:

- **FastAPI** – serves the trained machine learning model via a REST API.
- **Streamlit** – provides a simple web interface for users.

## Folder structure

```text
deployment/
├── airline_pipeline.pkl      # trained sklearn pipeline
├── predict.py                # prediction logic
├── api.py                    # FastAPI application
├── app.py                    # Streamlit application
├── requirements.txt          # deployment dependencies
└── README.md
```

## Install dependencies

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r deployment/requirements.txt
```

## Run FastAPI

Start the API server:

```bash
python deployment/api.py
```

or

```bash
uvicorn deployment.api:app --reload
```

The API will be available at:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

## Run Streamlit

Open a new terminal and run:

```bash
streamlit run deployment/app.py
```

The Streamlit application will open in your browser.

## Features

The application allows users to:

- enter an airline review;
- predict whether the review recommends the airline;
- display the model confidence score;
- highlight the most influential words contributing to the prediction.


## Live demo

Streamlit application:
https://airline-recommendation-app.onrender.com

FastAPI API:
https://airline-recommendation-api.onrender.com

## Project architecture

```text
Browser → Streamlit → FastAPI → Prediction
                           │
                           ▼
                 LinearSVC Pipeline
```

## Notes

The displayed confidence score is derived from the LinearSVC decision score using a sigmoid transformation. It provides
an estimate of prediction confidence and should not be interpreted as a calibrated probability.