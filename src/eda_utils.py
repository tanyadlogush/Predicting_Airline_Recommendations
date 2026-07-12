# Standard library
import warnings

# Third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from langdetect import detect

# Project modules
from src.config import PALETTE

warnings.filterwarnings('ignore', category=UserWarning)


def column_summary(df, col):
    """
    Prints basic statistics for a column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    col : str
        The name of the column to analyze.

    Notes
    -----
    - For categorical features, it shows the count of values, unique values, and missing values.
    - For numerical features, it additionally displays the skewness.
    """
    print('=' * 20, f'Column: {col}', '=' * 20)
    print(f'Total number of values: {df[col].shape[0]}')
    print(f'Number of unique values: {df[col].nunique()}')
    print(f'Number of missing values: {df[col].isna().sum()} ({df[col].isna().sum() / df[col].shape[0] * 100:.2f}%)')
    print(f'Data type: {df[col].dtype}')

    if pd.api.types.is_numeric_dtype(df[col]):
        print(f'Skewness: {df[col].skew().round(2)}')

    print('=' * 50)


def eda_category(df, col, target, plots=True):
    """
      Performs EDA for a categorical feature.

      Parameters
      ----------
      df : pandas.DataFrame
          The input dataframe.
      col : str
          The name of the categorical feature to analyze.
      target : str
          The name of the target variable in categorical form (e.g., "yes"/"no").
          Used for plotting and calculating percentage ratios.
      plots : bool, optional
          If True, builds and displays plots. Default is True.

      Notes
      -----
      - The target variable must be strictly categorical (e.g., "yes"/"no")
        to ensure clear legends and plots.
      - The function additionally outputs the value distribution (value_counts)
        and their percentages as part of the core logic.
    """
    # general information
    column_summary(df, col)

    # change NaN to "Missing values"
    value_counts_df = pd.DataFrame({
        'value_counts': df[col].fillna('Missing values').value_counts(),
        'value_percentage': df[col].fillna('Missing values').value_counts(normalize=True).round(4) * 100
    })

    # reorder the index
    idx = value_counts_df.index.tolist()
    if 'Missing values' in idx:
        idx = ['Missing values'] + [x for x in idx if x != 'Missing values']
        value_counts_df = value_counts_df.loc[idx]

    display(value_counts_df.T)
    print('=' * 50)

    # visualization
    if plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

        # histogram
        sns.countplot(
            data=df,
            x=col,
            hue=target,
            palette=PALETTE,
            ax=axes[0]
        )
        axes[0].set_title(f'Countplot: {col}')
        axes[0].grid(axis='y', alpha=0.7)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title=target)

        # barplot
        percent_df = df.groupby(col)[target].value_counts(normalize=True).unstack() * 100
        percent_df.plot(
            kind='bar',
            color=[PALETTE['no'], PALETTE['yes']],
            ax=axes[1]
        )

        axes[1].set_title('Recommended Distribution by Category')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.7)

        plt.tight_layout()
        plt.show()

        # Recommendation Rate
        yes_df = pd.crosstab(df[col], df[target], normalize='index') * 100

        plt.figure(figsize=(6, 4))
        yes_df['yes'].sort_values(ascending=False).plot(kind='bar', color=PALETTE['yes'])

        plt.title(f'Recommendation Rate by {col}', fontsize=13)
        plt.xlabel(col)
        plt.ylabel('P(Yes), %')
        plt.grid(axis='y', alpha=0.7)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


def eda_numeric(df, col, target_col, plots=True):
    """
      Performs EDA for a numerical feature.

      Parameters
      ----------
      df : pandas.DataFrame
          The input dataframe.
      col : str
          The name of the numerical feature to analyze.
      target_col : str, optional
          The name of the target variable in numerical form (e.g., 0/1).
          Used for calculating medians and plotting distribution graphs.
      plots : bool, optional
          If True, builds and displays plots. Default is True.

      Notes
      -----
      - The target variable here must be strictly numerical (0/1)
        to correctly calculate medians and statistics.
      - The function additionally estimates the number of outliers using the IQR rule
        and prints descriptive statistics—this block is used exclusively for numerical features.
      """

    # general information
    column_summary(df, col)

    # outliers
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

    # potential outliers (IQR rule)
    print(f'Outliers: {len(outliers)} ({round(len(outliers) / len(df[col]) * 100, 2)}%)')

    # medians no/yes
    median_0 = df[df[target_col] == 0][col].median()
    median_1 = df[df[target_col] == 1][col].median()
    print(f'Median for "no": {median_0}')
    print(f'Median for "yes": {median_1}')
    print('=' * 50)

    print(df[col].describe().round(2))
    print('=' * 50)

    # visualization
    if plots:
        labels = {0: 'no', 1: 'yes'}

        target_series = df[target_col].map(labels)

        plt.figure(figsize=(8, 8))

        # histogram
        plt.subplot(2, 1, 1)
        sns.histplot(data=df, x=col, hue=target_series, kde=True, palette=PALETTE)
        plt.title(f'Histogram of {df[col].name}')
        plt.grid()

        # boxplot
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x=target_series, y=col, hue=target_series,
                    palette=PALETTE)
        plt.title(f'Box Plot of {df[col].name}')
        plt.grid()

        # violinplot
        plt.subplot(2, 2, 4)
        sns.violinplot(data=df, x=col, y=target_series, hue=target_series, palette=PALETTE)
        plt.title(f'Violin Plot of {df[col].name}')

        plt.tight_layout()
        plt.grid()
        plt.show()


def detect_language(text):
    """
    Detects the language of a text.

    Returns "unknown" if language detection fails.
    """
    try:
        return detect(str(text))
    except Exception:
        return 'unknown'
