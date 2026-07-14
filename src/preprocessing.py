import re

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================ class TextCleaner ============================
class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans text data by removing noise and normalizing text format.
    """
    def fit(self, X, y=None):
        return self

    def _remove_html(self, text):
        return re.sub(r'<[^>]+>', ' ', text)

    def _remove_urls(self, text):
        return re.sub(r'https?://\S+|www\.\S+', ' ', text, flags=re.IGNORECASE)

    def _normalize_apostrophes(self, text):
        return text.replace('’', "'").replace('‘', "'").replace('`', "'")

    def _normalize_quotes(self, text):
        return (
            text.replace('“', '"')
            .replace('”', '"')
            .replace('„', '"')
        )

    def _normalize_repeated_punctuation(self, text):
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r':\)+', ':)', text)
        text = re.sub(r':\(+', ':(', text)
        return text

    def _normalize_whitespace(self, text):
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def _remove_standalone_punctuation(self, text):
        tokens = text.split()

        tokens = [
            token for token in tokens
            if not (
                    len(token) == 1
                    and not token.isalnum()
                    and token not in ['?', '!', ':']
            )
        ]

        return ' '.join(tokens)

    def transform(self, X):

        if isinstance(X, pd.DataFrame):
            # ColumnTransformer passes text columns as a single-column DataFrame
            text = X.iloc[:, 0]
        # elif isinstance(X, pd.Series):    # якщо (при побудові ColumnTransformer) виявиться, що версія sklearn передає ndarray
        #   text = X
        else:
            text = X

        text = text.copy().fillna('')

        text = text.apply(self._remove_html)
        text = text.apply(self._remove_urls)
        text = text.apply(self._normalize_apostrophes)
        text = text.apply(self._normalize_quotes)
        text = text.apply(self._normalize_repeated_punctuation)
        text = text.apply(self._normalize_whitespace)
        text = text.apply(self._remove_standalone_punctuation)

        return text

# functions
# ============================ remove_template_title ============================
def remove_template_title(title):
    """
    Removes template review titles of the form '<Airline> customer review'.
    """
    if pd.isna(title):
        return ''

    if re.search(r'customer review$', title, flags=re.IGNORECASE):
        return ''

    return title

# ============================ prepare_data ============================
def prepare_data(df, cols_to_drop):
    """
    Prepares the dataset by removing unnecessary columns
    and cleaning review titles.
    """
    df = df.copy()

    # remove unused features
    df = df.drop(columns=cols_to_drop)

    # remove template titles
    df['Review_Title'] = df['Review_Title'].apply(remove_template_title)

    return df

# ============================ prepare_target ============================
def prepare_target(df):
    """
    Converts the target variable from categorical labels
    to binary numerical values.
    """
    df = df.copy()

    df['Recommended_num'] = df['Recommended'].map({'yes': 1, 'no': 0})

    return df

# ============================ prepare_features ============================
def prepare_features(
    df,
    num_cols=None,
    cat_cols=None,
    text_source='review'
):
    """
    Prepares input features and target variable for model training.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe after data and target preprocessing.
    num_cols : list, optional
        List of numerical feature names.
    cat_cols : list, optional
        List of categorical feature names.
    text_source : {'review', 'title', 'review+title'}, default='review'
        Specifies the text source used to create the Review_Text feature:
        - 'review' : Use the review text only.
        - 'title' : Use the review title only.
        - 'review+title' : Combine the review title and review text.

    Returns
    -------
    X : pandas.DataFrame
        Input features.
    y : pandas.Series
        Target variable.
    """

    df = df.copy()

    if num_cols is None:
        num_cols = []
    elif not isinstance(num_cols, list):
        raise TypeError(
            f'num_cols must be a list, got {type(num_cols).__name__}.'
        )

    if cat_cols is None:
        cat_cols = []
    elif not isinstance(cat_cols, list):
        raise TypeError(
            f'cat_cols must be a list, got {type(cat_cols).__name__}.'
        )

    if text_source == 'review':
        df['Review_Text'] = df['Review']

    elif text_source == 'title':
        df['Review_Text'] = df['Review_Title']

    elif text_source == 'review+title':
        df['Review_Text'] = (
                df['Review_Title'].fillna('') + ' ' +
                df['Review'].fillna('')
        ).str.strip()

    else:
        raise ValueError(
            "text_features must be 'review', 'title' or 'review+title'."
        )

    input_cols = num_cols + cat_cols + ['Review_Text']

    missing_cols = [
    col for col in input_cols + ['Recommended_num']
    if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f'Missing columns: {missing_cols}'
        )

    X = df[input_cols].copy()
    y = df['Recommended_num']

    return X, y

# ============================ create_preprocessor ============================
def create_preprocessor(num_cols=None, cat_cols=None):
    """
    Creates a preprocessing pipeline for numerical, categorical,
    and text features.
    """

    if num_cols is None:
        num_cols = []
    elif not isinstance(num_cols, list):
        raise TypeError(
            f'num_cols must be a list, got {type(num_cols).__name__}.'
        )

    if cat_cols is None:
        cat_cols = []
    elif not isinstance(cat_cols, list):
        raise TypeError(
            f'cat_cols must be a list, got {type(cat_cols).__name__}.'
        )

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
    ])

    categorical_transformer = Pipeline([
        ('imputer',
         SimpleImputer(strategy='constant',
                       fill_value='Unknown')),
        ('encoder',
         OneHotEncoder(handle_unknown='ignore'))
    ])

    text_transformer = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        ))
    ])

    preprocessor = ColumnTransformer([

        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols),
        ('text', text_transformer, 'Review_Text')
    ])

    return preprocessor

# ============================ create_pipeline ============================
def create_pipeline(model, num_cols=None, cat_cols=None):
    """
    Create a machine learning pipeline that combines
    preprocessing and a classification model.
    """

    preprocessor = create_preprocessor(
        num_cols,
        cat_cols
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline
