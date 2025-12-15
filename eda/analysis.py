import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


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


def detect_outliers_iforest(df, column, contamination=0.05, random_state=42):
    series = df[column]
    valid_mask = series.notna()
    values = series[valid_mask].to_numpy().reshape(-1, 1)

    if len(values) == 0:
        return {'outliers_count': 0, 'lower_bound': None, 'upper_bound': None, 'outliers_data': None}

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    labels = model.fit_predict(values)
    outlier_indices = series[valid_mask].index[labels == -1]
    outliers = df.loc[outlier_indices]

    return {
        'outliers_count': len(outliers),
        'lower_bound': None,
        'upper_bound': None,
        'outliers_data': outliers if len(outliers) > 0 else None
    }


def get_outliers_summary(df, method='iqr', contamination=0.05):
    numeric_cols = df.select_dtypes(include='number').columns
    outliers_summary = {}

    for col in numeric_cols:
        if method == 'iforest':
            outliers_info = detect_outliers_iforest(df, col, contamination=contamination)
        else:
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
                valid_indices = [idx for idx in outlier_info['outliers_data'].index if idx in df.index]
                for idx in valid_indices:
                    all_outliers.append((idx, col, outlier_info['outliers_data'].loc[idx, col]))

    return all_outliers


def process_outliers(df, column, method, detect_method='iqr', contamination=0.05, n_neighbors=5):
    df_updated = df.copy()

    if detect_method == 'iforest':
        outlier_info = detect_outliers_iforest(df_updated, column, contamination=contamination)
    else:
        outlier_info = detect_outliers_iqr(df_updated, column)

    if outlier_info['outliers_count'] == 0 or outlier_info['outliers_data'] is None:
        return df_updated, 0

    processed_count = outlier_info['outliers_count']

    if method in ('iqr', 'clip', 'cap'):
        # Замена на границы, если они определены
        lower_bound = outlier_info['lower_bound']
        upper_bound = outlier_info['upper_bound']
        if lower_bound is None or upper_bound is None:
            return df_updated, 0
        df_updated[column] = df_updated[column].clip(lower=lower_bound, upper=upper_bound)

    elif method == 'median':
        # Замена на медиану
        median_val = df_updated[column].median()
        df_updated.loc[outlier_info['outliers_data'].index, column] = median_val

    elif method == 'mean':
        # Замена на среднее
        mean_val = df_updated[column].mean()
        df_updated.loc[outlier_info['outliers_data'].index, column] = mean_val

    elif method == 'knn':
        # Замена на значение из ближайших соседей
        numeric_cols = df_updated.select_dtypes(include=[np.number]).columns.tolist()
        
        # Убираем текущий столбец и столбцы с пропусками
        feature_cols = [col for col in numeric_cols if col != column and df_updated[col].notna().sum() > 0]
        
        if len(feature_cols) == 0:
            # Если нет других признаков, используем медиану
            median_val = df_updated[column].median()
            df_updated.loc[outlier_info['outliers_data'].index, column] = median_val
        else:
            # Подготовка данных: только строки без пропусков в feature_cols
            valid_mask = df_updated[feature_cols].notna().all(axis=1)
            valid_data = df_updated[valid_mask].copy()
            
            if len(valid_data) < n_neighbors + 1:
                # Если недостаточно данных, используем медиану
                median_val = df_updated[column].median()
                df_updated.loc[outlier_info['outliers_data'].index, column] = median_val
            else:
                # Разделяем на обучающие (без выбросов) и тестовые (выбросы) данные
                outlier_indices = outlier_info['outliers_data'].index
                train_mask = valid_mask & ~df_updated.index.isin(outlier_indices)
                train_data = df_updated[train_mask]
                
                if len(train_data) < n_neighbors:
                    # Если недостаточно обучающих данных, используем медиану
                    median_val = df_updated[column].median()
                    df_updated.loc[outlier_indices, column] = median_val
                else:
                    # Масштабирование признаков
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(train_data[feature_cols])
                    X_outliers = scaler.transform(df_updated.loc[outlier_indices, feature_cols])
                    y_train = train_data[column].values
                    
                    # Обучение KNN регрессора
                    knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(train_data)), weights='distance')
                    knn.fit(X_train, y_train)
                    
                    # Предсказание для выбросов
                    predictions = knn.predict(X_outliers)
                    df_updated.loc[outlier_indices, column] = predictions

    elif method == 'remove':
        # Удаление строк с выбросами
        df_updated = df_updated.drop(outlier_info['outliers_data'].index)

    return df_updated, processed_count


def has_numeric_missing(df):
    return df.select_dtypes(include='number').isna().any().any()


def has_categorical_missing(df):
    cat_cols = df.select_dtypes(include='object').columns
    return df[cat_cols].isna().any().any() if len(cat_cols) > 0 else False


def fill_numeric_missing(df, method='median', n_neighbors=5, random_state=42):
    df_filled = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) == 0:
        return df_filled

    if method == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
        return df_filled

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
    df_filled = df.copy()
    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        if df_filled[col].isna().any():
            if method == 'unknown':
                df_filled[col].fillna('Unknown', inplace=True)
            elif method in ('mode', 'most_frequent'):
                mode_val = df_filled[col].mode()
                if not mode_val.empty:
                    df_filled[col].fillna(mode_val[0], inplace=True)

    return df_filled


def remove_missing_rows(df):
    initial_rows = len(df)
    df_clean = df.dropna()
    removed_rows = initial_rows - len(df_clean)
    return df_clean, removed_rows


def split_name_column(df, name_col='name', brand_col='brand', model_col='model'):
    df_split = df.copy()

    if name_col not in df_split.columns:
        return df_split

    # Находим позицию столбца name
    name_position = df_split.columns.get_loc(name_col)

    # Разделяем name на brand и model
    df_split[brand_col] = df_split[name_col].str.split().str[0]
    df_split[model_col] = df_split[name_col].str.split(n=1).str[1].fillna('')

    # Удаляем исходный столбец name
    df_split = df_split.drop(columns=[name_col])

    # Переставляем столбцы: brand и model на место name
    cols = df_split.columns.tolist()
    # Удаляем brand и model из текущих позиций
    cols.remove(brand_col)
    cols.remove(model_col)
    # Вставляем brand и model на место name
    cols.insert(name_position, brand_col)
    cols.insert(name_position + 1, model_col)

    # Переупорядочиваем DataFrame
    df_split = df_split[cols]

    return df_split


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