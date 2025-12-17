import pandas as pd
import numpy as np
import streamlit as st
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


def get_outliers_summary(df, method='iqr'):
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
                valid_indices = [idx for idx in outlier_info['outliers_data'].index if idx in df.index]
                for idx in valid_indices:
                    all_outliers.append((idx, col, outlier_info['outliers_data'].loc[idx, col]))

    return all_outliers


def process_outliers(df, column, method, detect_method='iqr', n_neighbors=5):
    df_updated = df.copy()
    filled_positions = []

    outlier_info = detect_outliers_iqr(df_updated, column)

    if outlier_info['outliers_count'] == 0 or outlier_info['outliers_data'] is None:
        return df_updated, 0, []

    processed_count = outlier_info['outliers_count']
    outlier_indices = outlier_info['outliers_data'].index
    
    # Сохраняем старые значения для истории
    old_values = df_updated.loc[outlier_indices, column].copy()

    if method in ('iqr', 'clip', 'cap'):
        lower_bound = outlier_info['lower_bound']
        upper_bound = outlier_info['upper_bound']
        if lower_bound is None or upper_bound is None:
            return df_updated, 0, []
        
        df_updated[column] = df_updated[column].clip(lower=lower_bound, upper=upper_bound)

    elif method == 'median':
        median_val = df_updated[column].median()
        df_updated.loc[outlier_indices, column] = median_val

    elif method == 'mean':
        mean_val = df_updated[column].mean()
        df_updated.loc[outlier_indices, column] = mean_val

    elif method == 'knn':
        numeric_cols = df_updated.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != column and df_updated[col].notna().sum() > 0]
        
        if len(feature_cols) == 0 or len(df_updated) < n_neighbors + 1:
            median_val = df_updated[column].median()
            df_updated.loc[outlier_indices, column] = median_val
        else:
            valid_mask = df_updated[feature_cols].notna().all(axis=1)
            train_mask = valid_mask & ~df_updated.index.isin(outlier_indices)
            train_data = df_updated[train_mask]
            
            if len(train_data) < n_neighbors:
                 median_val = df_updated[column].median()
                 df_updated.loc[outlier_indices, column] = median_val
            else:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(train_data[feature_cols])
                X_outliers = scaler.transform(df_updated.loc[outlier_indices, feature_cols])
                y_train = train_data[column].values
                
                knn = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(train_data)), weights='distance')
                knn.fit(X_train, y_train)
                predictions = knn.predict(X_outliers)
                df_updated.loc[outlier_indices, column] = predictions

    elif method == 'remove':
        df_updated = df_updated.drop(outlier_indices)
        return df_updated, processed_count, []

    # Формируем историю изменений (если это не удаление)
    if method != 'remove':
        for idx in outlier_indices:
            # Проверяем, изменилось ли значение
            new_val = df_updated.loc[idx, column]
            old_val = old_values.loc[idx]
            
            if new_val != old_val:
                filled_positions.append({
                    'row': idx,
                    'column': column,
                    'old_value': old_val,
                    'new_value': new_val,
                    'row_display': idx + 1,
                    'fill_method': f'Outlier: {method}'
                })

    return df_updated, processed_count, filled_positions


def has_numeric_missing(df):
    return df.select_dtypes(include='number').isna().any().any()


def has_categorical_missing(df):
    cat_cols = df.select_dtypes(include='object').columns
    return df[cat_cols].isna().any().any() if len(cat_cols) > 0 else False


def fill_numeric_missing(df, method='median', n_neighbors=5, random_state=42):
    df_updated = df.copy()
    filled_positions = []
    
    numeric_cols = df_updated.select_dtypes(include='number').columns
    
    for col in numeric_cols:
        missing_mask = df_updated[col].isna()
        
        if missing_mask.any():
            if method == 'median':
                fill_value = df_updated[col].median()
                # Сохраняем старые значения
                old_values = df_updated.loc[missing_mask, col].copy()
                df_updated.loc[missing_mask, col] = fill_value
                
                for idx in df_updated[missing_mask].index:
                    filled_positions.append({
                        'row': idx,
                        'column': col,
                        'old_value': old_values.loc[idx],
                        'new_value': fill_value,
                        'row_display': idx + 1
                    })
                    
            elif method == 'mean':
                fill_value = df_updated[col].mean()
                old_values = df_updated.loc[missing_mask, col].copy()
                df_updated.loc[missing_mask, col] = fill_value
                
                for idx in df_updated[missing_mask].index:
                    filled_positions.append({
                        'row': idx,
                        'column': col,
                        'old_value': old_values.loc[idx],
                        'new_value': fill_value,
                        'row_display': idx + 1
                    })
                    
            elif method == 'knn':                
                temp_df = df_updated[numeric_cols].copy()
                old_values = df_updated.loc[missing_mask, col].copy()
                
                imputer = KNNImputer(n_neighbors=n_neighbors)
                temp_df_filled = pd.DataFrame(
                    imputer.fit_transform(temp_df),
                    columns=temp_df.columns,
                    index=temp_df.index
                )
                
                for idx in df_updated[missing_mask].index:
                    new_value = temp_df_filled.loc[idx, col]
                    df_updated.loc[idx, col] = new_value
                    
                    filled_positions.append({
                        'row': idx,
                        'column': col,
                        'old_value': old_values.loc[idx],
                        'new_value': new_value,
                        'row_display': idx + 1
                    })
    
    return df_updated, filled_positions


def fill_categorical_missing(df, method='unknown'):
    df_updated = df.copy()
    filled_positions = []
    
    categorical_cols = df_updated.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        missing_mask = df_updated[col].isna()
        
        if missing_mask.any():
            old_values = df_updated.loc[missing_mask, col].copy()  # Сохраняем старые значения
            
            if method == 'unknown':
                df_updated.loc[missing_mask, col] = 'Unknown'
            elif method == 'mode':
                mode_value = df_updated[col].mode()[0] if not df_updated[col].mode().empty else 'Unknown'
                df_updated.loc[missing_mask, col] = mode_value
            
            for idx in df_updated[missing_mask].index:
                filled_positions.append({
                    'row': idx,
                    'column': col,
                    'old_value': old_values.loc[idx],  # Сохраняем старое значение
                    'new_value': df_updated.loc[idx, col],
                    'row_display': idx + 1
                })
    
    return df_updated, filled_positions


def remove_missing_rows(df):
    df_updated = df.copy()
    
    # Находим строки с пропусками
    missing_mask = df_updated.isna().any(axis=1)
    removed_count = missing_mask.sum()
    
    # Удаляем строки
    df_updated = df_updated.dropna()
    
    return df_updated, removed_count


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


def generate_insights(df):
    info = dataset_info(df)
    rows, cols = info["shape"]
    missing = info["missing_values"]

    numeric_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(include='object').columns

    insights = []

    # Общая структура
    insights.append(f"Количество строк: {rows}")
    insights.append(f"Количество столбцов: {cols}")
    insights.append(f"Числовых признаков: {len(numeric_cols)}")
    insights.append(f"Категориальных признаков: {len(categorical_cols)}")

    # Пропуски
    missing_cols = {k: v for k, v in missing.items() if v > 0}
    total_missing = sum(missing.values())

    if not missing_cols:
        insights.append("Пропущенные значения отсутствуют")
    else:
        percent = round(total_missing / (rows * cols) * 100, 2)
        insights.append(
            f"Обнаружено {total_missing} пропущенных значений "
            f"в {len(missing_cols)} столбцах ({percent}%)"
        )

    # Сильные корреляции
    corr = correlation_matrix(df)
    strong_corr = []

    if corr is not None and not corr.empty:
        for i in corr.columns:
            for j in corr.columns:
                if i < j and abs(corr.loc[i, j]) > 0.6:
                    strong_corr.append(f"{i} – {j} ({corr.loc[i, j]:.2f})")

    if strong_corr:
        insights.append(
            "Обнаружены сильные корреляции между признаками: "
            + ", ".join(strong_corr)
        )

    if 'name' in df.columns or 'CarName' in df.columns:
        insights.append(
            "Обнаружен столбец с названием автомобиля — возможна декомпозиция на бренд и модель"
        )

    return insights


def get_change_history_df(df, history_data):
    if not history_data:
        return None
        
    all_changes_flat = []
    
    for batch in history_data:
        if batch:
            for pos in batch:
                # Получаем индекс строки
                row_idx = pos.get('row', 0)
                
                # Формируем информацию об идентификации (Имя или Бренд/Модель)
                ident_data = {}
                
                # Проверяем наличие разделенных столбцов brand и model
                if 'brand' in df.columns and 'model' in df.columns:
                    try:
                        ident_data['Brand'] = df.iloc[row_idx]['brand'] if row_idx < len(df) else 'Удалена'
                        ident_data['Model'] = df.iloc[row_idx]['model'] if row_idx < len(df) else 'Удалена'
                    except:
                        ident_data['Brand'] = '-'
                        ident_data['Model'] = '-'
                # Проверяем наличие обычного столбца name
                elif 'name' in df.columns:
                    try:
                        ident_data['Name'] = df.iloc[row_idx]['name'] if row_idx < len(df) else 'Удалена'
                    except:
                        ident_data['Name'] = '-'
                # Проверяем наличие CarName
                elif 'CarName' in df.columns:
                    try:
                        ident_data['Name'] = df.iloc[row_idx]['CarName'] if row_idx < len(df) else 'Удалена'
                    except:
                        ident_data['Name'] = '-'
                
                # Обработка старого значения
                old_val = pos.get('old_value', 'Неизвестно')
                if pd.isna(old_val):
                    old_val = "NaN"
                
                entry = {
                    '№ строки': pos.get('row_display', row_idx + 1),
                    **ident_data,
                    'Столбец': pos.get('column', ''),
                    'Старое значение': old_val,
                    'Новое значение': pos.get('new_value', ''),
                    'Метод': pos.get('fill_method', '-')
                }
                all_changes_flat.append(entry)
    
    if all_changes_flat:
        history_df = pd.DataFrame(all_changes_flat)
        
        # Определяем порядок столбцов
        base_cols = ['№ строки']
        if 'Brand' in history_df.columns and 'Model' in history_df.columns:
            base_cols.extend(['Brand', 'Model'])
        elif 'Name' in history_df.columns:
            base_cols.append('Name')
            
        base_cols.extend(['Столбец', 'Старое значение', 'Новое значение', 'Метод'])
        final_cols = [c for c in base_cols if c in history_df.columns]
        history_df = history_df[final_cols]
        
        # Разворачиваем (новые сверху)
        return history_df.iloc[::-1].reset_index(drop=True)
        
    return None

def display_change_history(df, history_data, title="История изменений"):
    history_df = get_change_history_df(df, history_data)
    
    if history_df is not None:
        st.write("---")
        st.write(f"#### {title}")
        st.dataframe(history_df, use_container_width=True)