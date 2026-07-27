import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.preprocessing import TextCleaner


MODEL_PATH = Path(__file__).parent / 'airline_pipeline.pkl'
model = joblib.load(MODEL_PATH)


def sigmoid(x: float) -> float:
    # convert decision function score to probability using sigmoid
    return 1 / (1 + np.exp(-x))


def get_top_features(review_text: str, prediction: int, top_n: int = 3) -> list[tuple[str, float]]:
    """Return the most influential words and phrases for the prediction."""

    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['model']

    df = pd.DataFrame({'Review_Text': [review_text]})
    tfidf_matrix = preprocessor.transform(df)

    tfidf_vectorizer = (
        preprocessor.named_transformers_['text']
        .named_steps['tfidf']
    )

    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]

    non_zero_indices = tfidf_matrix.nonzero()[1]

    contributions = []

    for idx in non_zero_indices:
        contribution = tfidf_matrix[0, idx] * coefficients[idx]

        if (prediction == 1 and contribution > 0) or (
                prediction == 0 and contribution < 0
        ):
            contributions.append((feature_names[idx], float(contribution)))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    explanation_stop_words = set(ENGLISH_STOP_WORDS) - {
        'not',
        'no',
        'nor',
        'never'
    }

    explanation_stop_words.update({
        'airline', 'flight', 'airport', 'aircraft', 'plane',
        'passenger', 'passengers', 'customer', 'customers',
        'just', 'yet'
    })

    candidates = contributions[:100]

    selected = []
    used_words = set()

    # prefer bigrams
    for feature, score in candidates:
        words = feature.split()

        if len(words) < 2:
            continue

        if (
                words[0] in explanation_stop_words
                or words[-1] in explanation_stop_words
        ):
            continue

        if any(word in used_words for word in words):
            continue

        selected.append((feature, score))
        used_words.update(words)

        if len(selected) == top_n:
            return selected

    # add unigrams
    for feature, score in candidates:
        words = feature.split()

        if len(words) != 1:
            continue

        word = words[0]

        if len(word) < 3:
            continue

        if word in explanation_stop_words:
            continue

        if word in used_words:
            continue

        selected.append((feature, score))
        used_words.add(word)

        if len(selected) == top_n:
            break

    return selected


def predict_recommendation(review_text: str) -> tuple[int, float, list[tuple[str, float]]]:
    # predict recommendation status, confidence score and top matching words
    df = pd.DataFrame({'Review_Text': [review_text]})

    # distance to decision boundary
    decision_score = model.decision_function(df)[0]

    # probability for positive class (1)
    prob_positive = sigmoid(decision_score)

    prediction = int(model.predict(df)[0])

    # confidence for predicted class
    confidence = prob_positive if prediction == 1 else (1 - prob_positive)

    # most influential words for the prediction
    top_features = get_top_features(review_text, prediction, top_n=3)

    return prediction, float(confidence), top_features