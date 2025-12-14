import streamlit as st
import pandas as pd
import hashlib
from eda.data_loader import load_data
from eda.analysis import (
    dataset_info, get_missing_rows, numerical_stats,
    get_categorical_analysis, get_outliers_summary, get_group_summary,
    get_all_outliers, process_outliers, has_numeric_missing, has_categorical_missing,
    fill_numeric_missing, fill_categorical_missing, remove_missing_rows, split_name_column
)
from eda.plots import (
    plot_missing, plot_bar,
    plot_correlation_heatmap, plot_scatter, plot_numerical_histograms,
    plot_all_boxplots, plot_hist_by_category
)
from eda.report_generator import render_summary_report
 
def _hash_dataframe(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@st.cache_data
def cached_get_outliers_summary(df_hash, method, contamination):
    df = st.session_state['df']
    return get_outliers_summary(df, method=method, contamination=contamination)

@st.cache_data
def cached_fill_numeric_missing(df_hash, method, n_neighbors, random_state):
    df = st.session_state['df']
    return fill_numeric_missing(df, method=method, n_neighbors=n_neighbors, random_state=random_state)

st.set_page_config(page_title="Автоматизированный EDA", layout="wide")
st.title("Автоматизированный модуль разведочного анализа данных (EDA)")

# Боковая панель
st.sidebar.title("Загрузка данных")
st.sidebar.markdown("### Выберите CSV-файл для анализа")

uploaded = True
uploaded = st.sidebar.file_uploader("Загрузить файл", type="csv")

st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

if uploaded:
    # Создаем вкладки для разных разделов анализа только после загрузки файла
    tab1, tab2, tab3, tab4, tab5, tab6= st.tabs([
        "Основная информация",
        "Визуализация пропусков",
        "Описательные статистики",
        "Анализ выбросов",
        "Анализ взаимосвязей",
        "Сводный отчёт"
    ])
    if st.sidebar.button("Вернуть исходные данные"):
        st.session_state['df'] = st.session_state['df_original'].copy()
        st.session_state['missing_filled'] = False
        st.session_state['outliers_filled'] = False
        st.cache_data.clear()
        st.rerun()

    # Проверяем, изменился ли файл (по имени и размеру)
    current_file_id = f"{uploaded.name}_{uploaded.size}"
    previous_file_id = st.session_state.get('current_file_id')

    if 'df' not in st.session_state or current_file_id != previous_file_id:
        # Загружаем новый файл
        df_initial = load_data(uploaded).reset_index(drop=True)
        # df_initial = pd.read_csv(r"D:\ПОЛИТЕХ\4 курс\Интеллектуальные системы и технологии\курсовая\eda_application\data\Automobile.csv")
        st.session_state['df'] = df_initial.copy()
        st.session_state['df_original'] = df_initial.copy()
        st.session_state['current_file_id'] = current_file_id
        st.session_state['missing_filled'] = False
        st.session_state['outliers_filled'] = False

    df = st.session_state['df']


    with tab1:
        st.header("Основная информация о датасете")
        
        # Разделение столбца name или CarName на brand и model
        column_to_split = None
        if 'name' in df.columns:
            column_to_split = 'name'
        elif 'CarName' in df.columns:
            column_to_split = 'CarName'
        
        if column_to_split and 'brand' not in df.columns and 'model' not in df.columns:
            st.write(f"#### Разделение столбца {column_to_split}")
            split_name = st.checkbox(
                f"Разделить столбец '{column_to_split}' на 'brand' (первое слово) и 'model' (остальные слова)",
                value=False,
                key=f"split_{column_to_split}_column"
            )

            if split_name:
                df_split = split_name_column(df, name_col=column_to_split)
                st.session_state['df'] = df_split
                st.success(f"Столбец '{column_to_split}' успешно разделен на 'brand' и 'model'!")
                st.rerun()
                
        st.write("#### Первые строки")
        st.dataframe(df.head())
        st.write("#### Последние строки")
        st.dataframe(df.tail())

        info = dataset_info(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Кол-во строк", info["shape"][0])
        col2.metric("Кол-во столбцов", info["shape"][1])

        st.write("#### Типы данных")
        dtypes_df = pd.DataFrame({
            'Столбец': list(info["dtypes"].keys()),
            'Тип данных': list(info["dtypes"].values())
        })
        st.dataframe(dtypes_df, use_container_width=True)

        # Подсчет количества числовых и категориальных колонок
        numeric_cols = df.select_dtypes(include='number').columns
        categorical_cols = df.select_dtypes(include='object').columns

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Числовых признаков", len(numeric_cols))
        with col2:
            st.metric("Категориальных признаков", len(categorical_cols))


    with tab2:
        st.header("Пропущенные значения")
        missing_df = pd.DataFrame({
            'Столбец': list(info["missing_values"].keys()),
            'Пропущенные значения': list(info["missing_values"].values())
        })
        st.dataframe(missing_df, use_container_width=True)

        st.write("#### Визуализация пропусков")
        st.pyplot(plot_missing(df))

        missing_filled = st.session_state.get('missing_filled', False)

        # Получаем актуальные строки с пропущенными значениями
        current_missing_rows = get_missing_rows(df)

        if not current_missing_rows.empty:
            with st.expander(f"Строки с пропущенными значениями ({len(current_missing_rows)} строк)"):
                st.dataframe(current_missing_rows, use_container_width=True)

        has_numeric_missing = has_numeric_missing(df)
        has_categorical_missing = has_categorical_missing(df)

        if has_numeric_missing or has_categorical_missing:
            st.write("#### Заполнение пропусков")

            # Ручное заполнение пропусков
            with st.expander("Ручное заполнение пропусков"):
                st.write("Выберите строку и столбец для заполнения пропуска:")

                # Находим все пропущенные значения
                missing_positions = []
                for row_idx, row in df.iterrows():
                    for col in df.columns:
                        if pd.isna(row[col]):
                            missing_positions.append((row_idx, col))

                if missing_positions:
                    # Выбор позиции для заполнения
                    position_options = [f"Строка {row+1}, столбец '{col}'" for row, col in missing_positions]
                    selected_position = st.selectbox(
                        "Выберите пропуск для заполнения:",
                        position_options,
                        key="manual_fill_position"
                    )

                    if selected_position:
                        # Получаем индекс выбранной позиции
                        position_idx = position_options.index(selected_position)
                        row_idx, col = missing_positions[position_idx]

                        # Определяем тип столбца и создаем соответствующий input
                        if df[col].dtype in ['int64', 'float64']:
                            # Числовой столбец
                            new_value = st.number_input(
                                f"Введите значение для {col} (строка {row_idx+1}):",
                                value=None,
                                key=f"manual_fill_{row_idx}_{col}"
                            )
                        else:
                            # Категориальный/текстовый столбец
                            # Получаем уникальные значения для подсказки
                            unique_values = df[col].dropna().unique().tolist()
                            new_value = st.selectbox(
                                f"Выберите значение для {col} (строка {row_idx+1}):",
                                options=[""] + unique_values,
                                key=f"manual_fill_{row_idx}_{col}"
                            )
                            # Если пользователь выбрал пустую строку, позволяем ввести свое значение
                            if new_value == "":
                                new_value = st.text_input(
                                    f"Или введите свое значение для {col} (строка {row_idx+1}):",
                                    key=f"manual_text_{row_idx}_{col}"
                                )

                        # Кнопка применения изменения
                        if st.button(f"Заполнить пропуск в строке {row_idx+1}, столбце '{col}'"):
                            if new_value is not None and str(new_value).strip() != "":
                                # Создаем копию датафрейма и заполняем пропуск
                                df_copy = df.copy()
                                df_copy.loc[row_idx, col] = new_value
                                st.session_state['df'] = df_copy
                                st.success(f"Пропуск в строке {row_idx+1}, столбце '{col}' заполнен значением: {new_value}")
                                st.rerun()
                            else:
                                st.error("Пожалуйста, введите значение для заполнения пропуска.")
                else:
                    st.info("Все пропуски уже заполнены!")

            # Автоматическое заполнение пропусков
            with st.expander("Автоматическое заполнение пропусков"):
                # Выбор метода обработки пропусков
                missing_method = st.radio(
                    "Метод обработки пропусков:",
                    ("Заполнение значениями", "Удаление строк с пропусками"),
                    key="missing_method"
                )

                if missing_method == "Заполнение значениями":
                    # Выбор метода для числовых пропусков
                    num_neighbors = 5
                    if has_numeric_missing:
                        num_method = st.radio(
                            "Метод заполнения числовых пропусков:",
                            ("Медиана", "Среднее", "KNN (ближайшие соседи)"),
                            key="num_fill_method"
                        )
                        if num_method == "KNN (ближайшие соседи)":
                            num_neighbors = st.slider("Число соседей для KNN", min_value=2, max_value=15, value=5, step=1)
                            st.info("Числовые пропуски будут заполнены на основе KNN по числовым признакам.")
                        elif num_method == "Медиана":
                            st.info("Числовые пропуски будут заполнены медианой.")
                        else:
                            st.info("Числовые пропуски будут заполнены средним значением.")
                    else:
                        num_method = "Медиана"  # значение по умолчанию

                    cat_method = "Неизвестно (Unknown)"  # значение по умолчанию

                    if has_categorical_missing:
                        cat_method = st.radio(
                            "Метод заполнения категориальных пропусков:",
                            ("Неизвестно (Unknown)", "Мода по столбцу"),
                            key="cat_fill_method"
                        )

                        if cat_method == "Мода по столбцу":
                            st.info("Пропуски в каждом категориальном столбце будут заполнены модой этого столбца.")
                        else:
                            st.info("Категориальные пропуски будут заполнены значением 'Unknown'.")
                    else:
                        st.info("Категориальные пропуски отсутствуют. Будут обработаны только числовые.")
                else:
                    st.info("Строки, содержащие пропуски, будут удалены из датасета.")

                # Кнопка заполнения
                if missing_method == "Заполнение значениями":
                    button_text = "Заполнить пропуски автоматически"
                else:
                    button_text = "Удалить строки с пропусками"

                if st.button(button_text):
                    # Создаем копию датафрейма для заполнения
                    df_updated = df.copy()

                    if missing_method == "Заполнение значениями":
                        # Преобразуем названия методов
                        num_fill_method_map = {
                            "Медиана": "median",
                            "Среднее": "mean",
                            "KNN (ближайшие соседи)": "knn"
                        }
                        num_fill_method = num_fill_method_map.get(num_method, 'median')
                        cat_fill_method = 'unknown' if cat_method == "Неизвестно (Unknown)" else 'mode'

                        # Заполняем числовые пропуски
                        if has_numeric_missing:
                            df_hash = _hash_dataframe(df)
                            df_updated = cached_fill_numeric_missing(
                                df_hash,
                                num_fill_method,
                                n_neighbors=num_neighbors,
                                random_state=42
                            )

                        # Заполняем категориальные пропуски
                        if has_categorical_missing:
                            df_updated = fill_categorical_missing(df_updated, cat_fill_method)

                        st.success("Все пропуски заполнены автоматически!")
                    else:
                        df_updated, removed_rows = remove_missing_rows(df_updated)

                        if removed_rows > 0:
                            st.success(f"Удалено {removed_rows} строк с пропусками. Новое количество строк: {len(df_updated)}")
                        else:
                            st.info("Не найдено строк с пропусками для удаления.")

                    st.session_state['df'] = df_updated
                    st.session_state['missing_filled'] = True
                    st.rerun()

        else:
            st.info("Все данные полные — заполнение пропусков не требуется.")


    with tab3:
        st.header("Описательные статистики")
        numeric_cols = df.select_dtypes(include='number').columns
        cat_cols = df.select_dtypes(include='object').columns

        st.write("#### Числовые признаки")
        st.dataframe(numerical_stats(df))

        # Гистограммы для числовых признаков
        if len(numeric_cols) > 0:
            st.write("#### Гистограммы числовых признаков")
            st.pyplot(plot_numerical_histograms(df))

        # Описание категориальных признаков
        if len(cat_cols) > 0:
            # Общая информация о категориальных данных
            cat_analysis = get_categorical_analysis(df)
            st.write("#### Категориальные признаки")
            for col, info in cat_analysis.items():
                with st.expander(f"{col}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Уникальных значений", info['unique_count'])
                    col2.metric("Всего значений", info['total_count'])
                    col3.metric("Пропусков", info['missing_count'])

                    st.write("**Наиболее частые значения:**")
                    most_common_df = pd.DataFrame({
                        'Значение': list(info['most_common'].keys()),
                        'Количество': list(info['most_common'].values())
                    })
                    st.dataframe(most_common_df, use_container_width=True)

                    if len(info['least_common']) <= 10:
                        st.write("**Наименее частые значения:**")
                        least_common_df = pd.DataFrame({
                            'Значение': list(info['least_common'].keys()),
                            'Количество': list(info['least_common'].values())
                        })
                        st.dataframe(least_common_df, use_container_width=True)

            # Визуализация категориальных данных
            st.write("#### Визуализация категориальных данных")
            column = st.selectbox("Выберите категориальную колонку для визуализации", cat_cols)
            st.plotly_chart(plot_bar(df, column))
        else:
            st.info("Нет категориальных столбцов")


    with tab4:
        st.header("Анализ выбросов")
        if len(numeric_cols) > 0:
            detection_choice = st.radio(
                "Метод поиска выбросов",
                ("IQR", "Isolation Forest"),
                key="outlier_detection_method"
            )

            contamination = 0.05

            if detection_choice == "Isolation Forest":
                contamination = st.slider("Доля выбросов (contamination)", 0.01, 0.2, 0.05, 0.01)

            detection_map = {
                "IQR": "iqr",
                "Isolation Forest": "iforest"
            }
            detect_method = detection_map[detection_choice]

            df_hash = hash(df.values.tobytes()) if not df.empty else 0
            cache_key = (df_hash, detect_method, contamination)

            # Кешируем через session_state (ручное кеширование)
            if "outliers_cache" not in st.session_state:
                st.session_state["outliers_cache"] = {}

            if cache_key not in st.session_state["outliers_cache"]:
                st.session_state["outliers_cache"][cache_key] = get_outliers_summary(
                    df, method=detect_method, contamination=contamination
                )

            df_hash = _hash_dataframe(df)
            outliers_summary = cached_get_outliers_summary(df_hash, detect_method, contamination)
            total_outliers = sum(info['outliers_count'] for info in outliers_summary.values())

            st.write("#### Визуализация выбросов")
            st.pyplot(plot_all_boxplots(df))

            # Таблица и детали — только если есть выбросы
            if total_outliers > 0:
                st.write(f"#### Обнаруженные выбросы ({detection_choice})")
                outliers_data = []
                for col, info in outliers_summary.items():
                    outliers_data.append({
                        'Признак': col,
                        'Количество выбросов': info['outliers_count'],
                        'Нижняя граница': round(info['lower_bound'], 3) if info['lower_bound'] is not None else "-",
                        'Верхняя граница': round(info['upper_bound'], 3) if info['upper_bound'] is not None else "-"
                    })
                st.dataframe(outliers_data, use_container_width=True)

                outlier_col = st.selectbox("Выберите признак для детального анализа выбросов", numeric_cols)

                # Детальный анализ — используем уже определённый outlier_col
                outlier_info = outliers_summary[outlier_col]

                if outlier_info['outliers_count'] > 0:
                    st.write(f"**Найдено выбросов: {outlier_info['outliers_count']}**")
                    if outlier_info['lower_bound'] is not None and outlier_info['upper_bound'] is not None:
                        st.write(f"Границы нормальных значений: [{outlier_info['lower_bound']:.3f}, {outlier_info['upper_bound']:.3f}]")
                    else:
                        st.write("Модель не задает явные границы, выбросы показаны списком.")
                    with st.expander("Показать строки с выбросами"):
                        st.dataframe(outlier_info['outliers_data'], use_container_width=True)
                else:
                    st.success(f"В признаке '{outlier_col}' выбросы не найдены")

            else:
                # Выбросов нет
                if st.session_state.get('outliers_filled', False):
                    st.success("Выбросы успешно обработаны!")
                else:
                    st.info("Выбросов в данных не обнаружено.")

            # Обработка выбросов
            st.write("#### Обработка выбросов")

            # Ручное заполнение выбросов
            with st.expander("Ручное заполнение выбросов"):
                # Находим все выбросы
                all_outliers = get_all_outliers(df, outliers_summary)

                if all_outliers:
                    st.write("Выберите выброс для замены:")

                    # Выбор выброса для замены
                    outlier_options = [f"Строка {row+1}, столбец '{col}', значение: {val}" for row, col, val in all_outliers]
                    selected_outlier = st.selectbox(
                        "Выберите выброс для замены:",
                        outlier_options,
                        key="manual_outlier_fill"
                    )

                    if selected_outlier:
                        # Получаем индекс выбранного выброса
                        outlier_idx = outlier_options.index(selected_outlier)
                        row_idx, col, current_val = all_outliers[outlier_idx]

                        # Ввод нового значения
                        new_value = st.number_input(
                            f"Введите новое значение для {col} (строка {row_idx+1}):",
                            value=None,
                            key=f"outlier_fill_{row_idx}_{col}"
                        )

                        # Кнопка применения изменения
                        if st.button(f"Заменить выброс в строке {row_idx+1}, столбце '{col}'"):
                            if new_value is not None:
                                # Создаем копию датафрейма и заменяем выброс
                                df_copy = df.copy()
                                df_copy.loc[row_idx, col] = new_value
                                st.session_state['df'] = df_copy
                                st.success(f"Выброс в строке {row_idx+1}, столбце '{col}' заменен с {current_val} на {new_value}")
                                st.rerun()
                            else:
                                st.error("Пожалуйста, введите числовое значение.")
                else:
                    st.info("Выбросов не найдено!")

            # Автоматическая обработка выбросов
            with st.expander("Автоматическая обработка выбросов"):
                # Проверяем наличие выбросов для обработки
                has_outliers_for_auto = any(outliers_summary[col]['outliers_count'] > 0 for col in numeric_cols)

                if has_outliers_for_auto:
                    st.write("Выберите метод обработки выбросов:")

                    # Выбор метода
                    n_neighbors = 5
                    if detect_method == "iforest":
                        outlier_method = st.radio(
                            "Метод обработки:",
                            ("Замена на медиану", "Замена на среднее", "KNN (ближайшие соседи)", "Удаление выбросов"),
                            key="auto_outlier_method"
                        )
                    else:
                        outlier_method = st.radio(
                            "Метод обработки:",
                            ("Замена на границы", "Замена на медиану", "Замена на среднее", "KNN (ближайшие соседи)", "Удаление выбросов"),
                            key="auto_outlier_method"
                        )

                    if outlier_method == "KNN (ближайшие соседи)":
                        n_neighbors = st.slider("Число соседей для KNN", min_value=2, max_value=15, value=5, step=1)
                        st.info("Выбросы будут заменены на значения, предсказанные по k ближайшим соседям на основе других признаков.")
                    elif outlier_method == "Замена на границы":
                        st.info("Выбросы будут заменены на ближайшие границы нормальных значений.")
                    elif outlier_method == "Замена на медиану":
                        st.info("Выбросы будут заменены на медиану столбца.")
                    elif outlier_method == "Замена на среднее":
                        st.info("Выбросы будут заменены на среднее значение столбца.")
                    else:
                        st.info("Строки с выбросами будут удалены.")

                    # Выбор области применения
                    process_all_columns = st.checkbox("Применить ко всем столбцам", value=False, key="process_all_columns")

                    if process_all_columns:
                        cols_to_process = numeric_cols.tolist()
                        button_text = f"Применить {outlier_method.lower()} ко всем столбцам"
                    else:
                        cols_to_process = [outlier_col]
                        button_text = f"Применить {outlier_method.lower()} к столбцу '{outlier_col}'"

                    # Кнопка применения
                    if st.button(button_text):
                        # Преобразуем название метода для функции
                        method_map = {
                            "Замена на границы": "clip",
                            "Замена на медиану": "median",
                            "Замена на среднее": "mean",
                            "KNN (ближайшие соседи)": "knn",
                            "Удаление выбросов": "remove"
                        }
                        method = method_map[outlier_method]

                        # Обрабатываем выбросы
                        df_updated = df.copy()
                        total_processed = 0

                        for col in cols_to_process:
                            col_updated, processed_count = process_outliers(
                                df_updated,
                                col,
                                method,
                                detect_method=detect_method,
                                contamination=contamination,
                                n_neighbors=n_neighbors
                            )
                            df_updated = col_updated
                            total_processed += processed_count

                        st.session_state['df'] = df_updated
                        st.session_state['outliers_filled'] = True

                        # Пересчитываем outliers_summary для всех методов
                        outliers_summary = get_outliers_summary(df_updated, method=detect_method, contamination=contamination)
                        total_outliers = sum(info['outliers_count'] for info in outliers_summary.values())

                        if total_processed > 0:
                            if outlier_method.startswith("Удаление выбросов"):
                                st.success(f"Удалено {total_processed} выбросов из {len(cols_to_process)} столбцов!")
                            else:
                                columns_text = "всех столбцов" if process_all_columns else f"столбца '{outlier_col}'"
                                st.success(f"Обработано {total_processed} выбросов методом '{outlier_method}' в {columns_text}")
                        else:
                            columns_text = "выбранных столбцах" if process_all_columns else f"столбце '{outlier_col}'"
                            st.info(f"В {columns_text} не найдено выбросов для обработки методом '{outlier_method}'")

                        st.rerun()
                else:
                    st.info("Выбросов для автоматической обработки не найдено!")
        else:
            st.info("Нет числовых столбцов для анализа выбросов")


    with tab5:
        st.header("Анализ взаимосвязей")

        if len(numeric_cols) > 1:
            st.write("#### Матрица корреляций")
            st.pyplot(plot_correlation_heatmap(df))

            st.write("#### Диаграмма рассеяния")
            col1 = st.selectbox("X", numeric_cols)
            col2 = st.selectbox("Y", numeric_cols)
            st.plotly_chart(plot_scatter(df, col1, col2))
        else:
            st.info("Недостаточно числовых столбцов для корреляционного анализа")

        # Сравнение распределений по категориям
        if len(numeric_cols) > 0 and len(cat_cols) > 0:
            st.write("#### Сравнение распределений по категориям")

            comp_numeric = st.selectbox("Выберите числовой признак", numeric_cols, key="comp_numeric")
            comp_category = st.selectbox("Выберите категориальный признак", cat_cols, key="comp_category")
            st.pyplot(plot_hist_by_category(df, comp_numeric, comp_category))

        # Сводные таблицы
        if len(cat_cols) > 0:
            st.write("#### Сводные таблицы по категориям")

            summary_col = st.selectbox("Выберите категорию для группировки", cat_cols, key="summary")
            summary = get_group_summary(df, summary_col)

            if summary:
                tab1, tab2, tab3, tab4 = st.tabs(["Средние значения", "Минимумы", "Максимумы", "Количество"])

                with tab1:
                    st.dataframe(summary['means'])

                with tab2:
                    st.dataframe(summary['mins'])

                with tab3:
                    st.dataframe(summary['maxs'])

                with tab4:
                    st.dataframe(summary['counts'])
            else:
                st.error("Ошибка при создании сводной таблицы")

    with tab6:
        st.header("Сводный отчёт по датасету")
        st.info("Автоматически сгенерированный отчёт с выводами и визуализациями.")

        # Получаем метод выбросов из session_state (если пользователь его менял)
        detect_choice = st.session_state.get('outlier_detection_method', 'IQR')
        contamination = st.session_state.get('contamination', 0.05)
        detect_method = 'iforest' if detect_choice == 'Isolation Forest' else 'iqr'

        with st.spinner("Генерация отчёта..."):
            render_summary_report(df, detect_method=detect_method, contamination=contamination)

        st.subheader("HTML-отчёт")

        if "summary_html" in st.session_state:
            st.download_button(
                "Скачать HTML-отчёт",
                data=st.session_state["summary_html"],
                file_name="eda_summary_report.html",
                mime="text/html"
            )



else:
    # Если файл не загружен, показываем инструкцию
    st.info("Для начала анализа данных загрузите CSV-файл через боковую панель.")
