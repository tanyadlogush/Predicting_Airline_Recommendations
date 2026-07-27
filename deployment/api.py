import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from deployment.predict import predict_recommendation


app = FastAPI(
    title='Airline Review Sentiment API',
    description='API for predicting airline recommendation based on review text',
    version='1.0.0'
)


class ReviewRequest(BaseModel):
    review_text: str


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    top_features: list[tuple[str, float]]


@app.get('/')
def root():
    return {'message': 'Airline Recommendation API is running'}


@app.post('/predict', response_model=PredictionResponse)
def predict(request: ReviewRequest):
    review_text = request.review_text.strip()
    if not review_text:
        raise HTTPException(
            status_code=400,
            detail='Review text cannot be empty.'
        )

    prediction, confidence, top_features = predict_recommendation(review_text)

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        top_features=top_features
    )


if __name__ == '__main__':
    import uvicorn


    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        'deployment.api:app',
        host='0.0.0.0',
        port=port,
        reload=False
    )