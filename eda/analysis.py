def correlation_matrix(df):
    return df.corr(numeric_only=True)

def dataset_info(df):
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict()
    }

def get_missing_rows(df):
    return df[df.isna().any(axis=1)]

def numerical_stats(df):
    return df.describe(include='number')

def categorical_stats(df):
    cat_cols = df.select_dtypes(include='object').columns
    result = {}

    for col in cat_cols:
        result[col] = {
            "unique_values": df[col].unique().tolist(),
            "unique_count": df[col].nunique()
        }

    return result

def get_categorical_analysis(df):
    cat_cols = df.select_dtypes(include='object').columns
    analysis = {}

    for col in cat_cols:
        unique_values = df[col].unique()
        value_counts = df[col].value_counts()
        analysis[col] = {
            'unique_count': len(unique_values),
            'total_count': len(df[col]),
            'missing_count': df[col].isnull().sum(),
            'most_common': value_counts.head(10).to_dict(),
            'least_common': value_counts.tail(5).to_dict() if len(value_counts) > 5 else value_counts.to_dict()
        }

    return analysis

def calculate_iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def detect_outliers_iqr(df, column):
    lower_bound, upper_bound = calculate_iqr_bounds(df[column])
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return {
        'outliers_count': len(outliers),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_data': outliers if len(outliers) > 0 else None
    }

def get_outliers_summary(df):
    numeric_cols = df.select_dtypes(include='number').columns
    outliers_summary = {}

    for col in numeric_cols:
        outliers_info = detect_outliers_iqr(df, col)
        outliers_summary[col] = outliers_info

    return outliers_summary

def get_all_outliers(df, outliers_summary):
    all_outliers = []
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if col in outliers_summary:
            outlier_info = outliers_summary[col]
            if outlier_info['outliers_count'] > 0 and outlier_info['outliers_data'] is not None:
                # Проверяем, что индексы все еще существуют в текущем датафрейме
                valid_indices = [idx for idx in outlier_info['outliers_data'].index if idx in df.index]
                for idx in valid_indices:
                    all_outliers.append((idx, col, outlier_info['outliers_data'].loc[idx, col]))

    return all_outliers

def process_outliers(df, column, method):
    df_updated = df.copy()
    outlier_info = detect_outliers_iqr(df_updated, column)

    if outlier_info['outliers_count'] == 0 or outlier_info['outliers_data'] is None:
        return df_updated, 0

    processed_count = outlier_info['outliers_count']

    if method == 'iqr':
        # Замена на границы IQR - используем clip для надежности
        lower_bound = outlier_info['lower_bound']
        upper_bound = outlier_info['upper_bound']
        df_updated[column] = df_updated[column].clip(lower=lower_bound, upper=upper_bound)

    elif method == 'median':
        # Замена на медиану
        median_val = df_updated[column].median()
        for idx in outlier_info['outliers_data'].index:
            df_updated.loc[idx, column] = median_val

    elif method == 'mean':
        # Замена на среднее
        mean_val = df_updated[column].mean()
        for idx in outlier_info['outliers_data'].index:
            df_updated.loc[idx, column] = mean_val

    elif method == 'remove':
        # Удаление строк с выбросами
        df_updated = df_updated.drop(outlier_info['outliers_data'].index)

    return df_updated, processed_count

def has_numeric_missing(df):
    """Проверяет наличие пропусков в числовых столбцах"""
    return df.select_dtypes(include='number').isna().any().any()

def has_categorical_missing(df):
    """Проверяет наличие пропусков в категориальных столбцах"""
    cat_cols = df.select_dtypes(include='object').columns
    return df[cat_cols].isna().any().any() if len(cat_cols) > 0 else False

def fill_numeric_missing(df, method='median'):
    """
    Заполняет пропуски в числовых столбцах

    Args:
        df: DataFrame
        method: 'median' или 'mean'

    Returns:
        DataFrame с заполненными пропусками
    """
    df_filled = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if df_filled[col].isna().any():
            if method == 'median':
                fill_val = df_filled[col].median()
            elif method == 'mean':
                fill_val = df_filled[col].mean()
            else:
                continue
            df_filled[col].fillna(fill_val, inplace=True)

    return df_filled

def fill_categorical_missing(df, method='unknown'):
    """
    Заполняет пропуски в категориальных столбцах

    Args:
        df: DataFrame
        method: 'unknown' или 'mode'

    Returns:
        DataFrame с заполненными пропусками
    """
    df_filled = df.copy()
    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        if df_filled[col].isna().any():
            if method == 'unknown':
                df_filled[col].fillna('Unknown', inplace=True)
            elif method == 'mode':
                mode_val = df_filled[col].mode()
                if not mode_val.empty:
                    df_filled[col].fillna(mode_val[0], inplace=True)

    return df_filled

def remove_missing_rows(df):
    """
    Удаляет все строки, содержащие хотя бы один пропуск

    Args:
        df: DataFrame

    Returns:
        tuple: (DataFrame без пропусков, количество удаленных строк)
    """
    initial_rows = len(df)
    df_clean = df.dropna()
    removed_rows = initial_rows - len(df_clean)
    return df_clean, removed_rows

def get_group_summary(df, group_by_col, agg_cols=None):
    if agg_cols is None:
        agg_cols = df.select_dtypes(include='number').columns.tolist()

    if group_by_col not in df.columns:
        return None

    # Средние значения
    means = df.groupby(group_by_col)[agg_cols].mean()

    # Минимальные и максимальные значения
    mins = df.groupby(group_by_col)[agg_cols].min()
    maxs = df.groupby(group_by_col)[agg_cols].max()

    # Количество наблюдений
    counts = df.groupby(group_by_col).size()

    return {
        'means': means,
        'mins': mins,
        'maxs': maxs,
        'counts': counts
    }