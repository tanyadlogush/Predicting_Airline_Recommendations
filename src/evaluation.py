import pandas as pd
import matplotlib.pyplot as plt

from IPython.core.display_functions import display
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve


# ============================ class ResultsTable ========================
# ========================================================================
class ResultsTable:
    """
    Manage a table of model evaluation results.

    The class stores evaluation metrics for multiple models and provides
    methods to add, update, remove, display, save, and load results.
    It is intended to simplify model comparison across different
    experiments.
    """
    def __init__(self):
        self.table = pd.DataFrame(columns=[
            'Model',
            'Configuration',
            'Train_F1',
            'Val_F1',
            'Train_AUROC',
            'Val_AUROC'
        ])

    def add_result(self, new_result: dict):
        new_row = pd.DataFrame([new_result])

        self.table = pd.concat(
            [self.table, new_row],
            ignore_index=True
        )

    def update_configuration(self, index: int, configuration: str):
        if not 0 <= index < len(self.table):
            raise IndexError(f'Row {index} does not exist.')

        self.table.loc[index, 'Configuration'] = configuration

    def remove_result(self, index):
        if not 0 <= index < len(self.table):
            raise IndexError(f'Row {index} does not exist.')

        self.table.drop([index], inplace=True)
        self.table.reset_index(drop=True, inplace=True)

    def display(self):
        pd.set_option('display.width', 150)
        pd.set_option('display.max_colwidth', 80)
        display(self.table)

    def save(self, path='../models/model_results.csv'):
        self.table.to_csv(path, index=False)

    def to_dataframe(self):
        return self.table.copy()

    @classmethod
    def load(cls, path='../models/model_results.csv'):
        results = cls()
        results.table = pd.read_csv(path)
        return results


# ============================ get_model_name ============================
# ========================================================================
def get_model_name(model):
    return model.named_steps['model'].__class__.__name__


# ============================ evaluate_model ============================
# ========================================================================
def evaluate_model(model, X_train, y_train, X_eval, y_eval, dataset_name='validation'):
    """
    Evaluate a trained classification pipeline on training and evaluation datasets.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained classification pipeline containing preprocessing steps
        and a final estimator with predict() and either predict_proba()
        or decision_function().

    X_train : pandas.DataFrame
        Training features.

    y_train : pandas.Series
        Training target values.

    X_eval : pandas.DataFrame
        Evaluation features (e.g. validation or test set).

    y_eval : pandas.Series
        Evaluation target values.

    dataset_name : str, default='validation'
        Name of the evaluation dataset used in printed reports.
        Typical values are 'validation' or 'test'.

    Returns
    -------
      dict
        Dictionary containing:
        - Model
        - Configuration
        - Train_F1
        - Val_F1
        - Train_AUROC
        - Val_AUROC
    """

    model_name = get_model_name(model)

    # predictions
    y_train_pred = model.predict(X_train)
    y_eval_pred = model.predict(X_eval)

    # prediction scores
    if hasattr(model, 'predict_proba'):
        y_train_score = model.predict_proba(X_train)[:, 1]
        y_eval_score = model.predict_proba(X_eval)[:, 1]

    elif hasattr(model, 'decision_function'):
        y_train_score = model.decision_function(X_train)
        y_eval_score = model.decision_function(X_eval)

    else:
        raise ValueError(
            f'{model_name} does not support either predict_proba() or decision_function().'
        )

    # metrics
    train_f1 = f1_score(y_train, y_train_pred)
    eval_f1 = f1_score(y_eval, y_eval_pred)

    train_auc = roc_auc_score(y_train, y_train_score)
    eval_auc = roc_auc_score(y_eval, y_eval_score)

    # reports
    print(f'\n=== {model_name} ===')
    print(f'Classification report ({dataset_name}):')
    print(classification_report(y_eval, y_eval_pred))

    print(f'Confusion matrix ({dataset_name}):')
    print(confusion_matrix(y_eval, y_eval_pred))

    print(f'Train F1: {train_f1:.4f}')
    print(f'{dataset_name.capitalize()} F1: {eval_f1:.4f}')
    print(f'Train AUROC: {train_auc:.4f}')
    print(f'{dataset_name.capitalize()} AUROC: {eval_auc:.4f}')

    return {
        'Model': model_name,
        'Configuration': '',
        'Train_F1': round(train_f1, 4),
        'Val_F1': round(eval_f1, 4),
        'Train_AUROC': round(train_auc, 4),
        'Val_AUROC': round(eval_auc, 4)
    }


# ============================ plot_roc_curve ============================
# ========================================================================
def plot_roc_curve(model, X_eval, y_eval):
    """
    Plot ROC curve for a classification model.
    """

    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_eval)[:, 1]
    else:
        y_score = model.decision_function(X_eval)

    auc_score = roc_auc_score(y_eval, y_score)
    model_name = get_model_name(model)

    fpr, tpr, _ = roc_curve(y_eval, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


