import re

import pandas as pd
import matplotlib.pyplot as plt

import warnings

from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, \
    accuracy_score


# ============================ get_model_name ============================
# ========================================================================
def get_model_name(model):
    return model.named_steps['model'].__class__.__name__


# ============================ evaluate_model ============================
# ========================================================================
def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluate a trained classification pipeline on training and validation data.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained classification pipeline containing preprocessing steps
        and a final estimator with predict() and predict_proba() methods.

    X_train : pandas.DataFrame
        Training features.

    y_train : pandas.Series
        Training target values.

    X_val : pandas.DataFrame
        Validation features.

    y_val : pandas.Series
        Validation target values.

    Returns
    -------
    dict
        Dictionary containing model name and evaluation metrics:
        Train F1-score, Validation F1-score,
        Train ROC-AUC, and Validation ROC-AUC.
    """

    model_name = get_model_name(model)

    # predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # predicted probabilities
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_val_proba = model.predict_proba(X_val)[:, 1]
    else:
        raise ValueError(f'{model_name} does not support predict_proba')

    # metrics
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)

    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    # додаткові звіти
    print(f'\n=== {model_name} ===')
    print('Classification report (validation):')
    print(classification_report(y_val, y_val_pred))

    print('Confusion matrix (validation):')
    print(confusion_matrix(y_val, y_val_pred))

    print(f'Train F1: {train_f1:.4f}')
    print(f'Validation F1: {val_f1:.4f}')
    print(f'Train AUROC: {train_auc:.4f}')
    print(f'Validation AUROC: {val_auc:.4f}')

    return {
        'Model': model_name,
        'Configuration': '',
        'Train_F1': round(train_f1, 4),
        'Val_F1': round(val_f1, 4),
        'Train_AUROC': round(train_auc, 4),
        'Val_AUROC': round(val_auc, 4)
    }


# ============================ plot_roc_curve ============================
# ========================================================================
def plot_roc_curve(model, X_val, y_val):
    """
    Plot ROC curve for a classification model.
    """

    y_proba = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_proba)
    model_name = get_model_name(model)

    fpr, tpr, _ = roc_curve(y_val, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()





# ============================   ============================
# ========================================================================


# ============================   ============================
# ========================================================================


# ============================   ============================
# ========================================================================
