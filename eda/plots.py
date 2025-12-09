import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from .analysis import correlation_matrix

def plot_missing(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(df.isnull(), cbar=False, ax=axes[0])
    axes[0].set_title("Тепловая карта пропусков")

    df.isnull().sum().plot(kind='bar', ax=axes[1])
    axes[1].set_title("Количество пропусков по признакам")
    axes[1].set_ylabel("Число пропусков")

    plt.tight_layout()
    return fig

def plot_numerical_histograms(df):
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        return None

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row, col_idx] if n_rows > 1 and n_cols > 1 else axes[i]
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'{col}')
        ax.set_xlabel(col)
        # ax.set_ylabel('Частота')

    # Убираем пустые subplot'ы
    for i in range(len(numeric_cols), n_rows * n_cols):
        row, col_idx = divmod(i, n_cols)
        if n_rows > 1 and n_cols > 1:
            fig.delaxes(axes[row, col_idx])

    plt.tight_layout()
    return fig

def plot_bar(df, column):
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = ['category', 'count']

    fig = px.bar(value_counts,
                 x='category', y='count',
                 labels={'category': 'Категория', 'count': 'Количество'})
    return fig

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix(df), annot=True, cmap="coolwarm", ax=ax)
    return fig

def plot_scatter(df, col1, col2):
    return px.scatter(df, x=col1, y=col2)

def plot_distribution_by_category(df, numeric_col, category_col, top_n=10):
    if category_col not in df.columns or numeric_col not in df.columns:
        return None

    # Берем топ-N категорий по частоте
    top_categories = df[category_col].value_counts().head(top_n).index
    filtered_df = df[df[category_col].isin(top_categories)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x=category_col, y=numeric_col, ax=ax)
    ax.set_title(f'Распределение {numeric_col} по {category_col} (топ-{top_n})')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    return fig

def plot_hist_by_category(df, numeric_col, category_col, top_n=5):
    if category_col not in df.columns or numeric_col not in df.columns:
        return None

    # Берем топ-N категорий по частоте
    top_categories = df[category_col].value_counts().head(top_n).index
    filtered_df = df[df[category_col].isin(top_categories)]

    fig, ax = plt.subplots(figsize=(12, 6))
    for category in top_categories:
        subset = filtered_df[filtered_df[category_col] == category]
        sns.histplot(subset[numeric_col], kde=True, label=category, ax=ax, alpha=0.7)

    ax.set_title(f'Распределение {numeric_col} по {category_col} (топ-{top_n})')
    ax.legend()
    plt.tight_layout()

    return fig

def plot_all_boxplots(df):
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) == 0:
        return None

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row, col_idx] if n_rows > 1 and n_cols > 1 else axes[i]
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(f'{col}')

    # Убираем пустые subplot'ы
    for i in range(len(numeric_cols), n_rows * n_cols):
        row, col_idx = divmod(i, n_cols)
        if n_rows > 1 and n_cols > 1:
            fig.delaxes(axes[row, col_idx])

    plt.tight_layout()
    return fig