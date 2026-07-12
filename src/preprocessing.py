import re

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def remove_template_title(title):
    """
    Removes template review titles of the form '<Airline> customer review'.
    """
    if pd.isna(title):
        return ''

    if re.search(r'customer review$', title, flags=re.IGNORECASE):
        return ''

    return title


def prepare_data(df, cols_to_drop):
    """
    Prepares the dataset by removing unnecessary columns,
    cleaning review titles, and creating combined review text.
    """
    df = df.copy()

    # remove unused features
    df = df.drop(columns=cols_to_drop)

    # remove template titles
    df['Review_Title'] = df['Review_Title'].apply(remove_template_title)

    # create combined text feature
    df['Review_Text'] = (
            df['Review_Title'].fillna('') + ' ' + df['Review'].fillna('')
    ).str.strip()

    return df


def prepare_target(df):
    """
    Converts the target variable from categorical labels
    to binary numerical values.
    """
    df = df.copy()

    df['Recommended_num'] = df['Recommended'].map({'yes': 1, 'no': 0})

    return df


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


def create_preprocessor(num_cols, cat_cols, text_col):
    """
    Creates a preprocessing pipeline for numerical, categorical,
    and text features.
    """

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
        ('text', text_transformer, text_col)
    ])

    return preprocessor
