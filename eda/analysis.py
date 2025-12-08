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

def fill_missing_values(df, cat_method='unknown', group_col=None):
    df = df.copy()
    filled_info = {}

    # Числовые колонки заполняются медианой
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            filled_info[col] = float(median_val)

    # Категориальные колонки
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isna().any():
            if cat_method == 'unknown':
                df[col].fillna('Unknown', inplace=True)
                filled_info[col] = 'Unknown'
            elif cat_method == 'mode':
                # Заполняем модой по всему столбцу
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                    filled_info[col] = mode_val[0]
                else:
                    df[col].fillna('Unknown', inplace=True)
                    filled_info[col] = 'Unknown'
            elif cat_method == 'group' and group_col and group_col in df.columns and group_col != col:
                # Заполняем модой внутри каждой группы
                def fill_by_group(series):
                    mode_val = series.mode()
                    if not mode_val.empty:
                        return mode_val[0]
                    return 'Unknown'  # на случай, если в группе все NaN
                
                # Группируем и заполняем
                df[col] = df.groupby(group_col)[col].transform(
                    lambda x: x.fillna(fill_by_group(x))
                )
                # Если после группировки остались NaN (например, все значения в группе были NaN)
                df[col].fillna('Unknown', inplace=True)
                filled_info[col] = f"Заполнено по группе '{group_col}'"
            else:
                # fallback
                df[col].fillna('Unknown', inplace=True)
                filled_info[col] = 'Unknown'

    return df, filled_info

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

def fill_outliers_iqr(df, columns=None):
    df_clean = df.copy()
    if columns is None:
        columns = df_clean.select_dtypes(include='number').columns

    for col in columns:
        lower_bound, upper_bound = calculate_iqr_bounds(df_clean[col])
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

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