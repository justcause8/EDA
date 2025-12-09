import streamlit as st
from eda.data_loader import load_data
from eda.analysis import (
    dataset_info, get_missing_rows, fill_missing_values, numerical_stats,
    get_categorical_analysis, get_outliers_summary, fill_outliers_iqr, get_group_summary
)
from eda.plots import (
    plot_missing, plot_bar,
    plot_correlation_heatmap, plot_scatter, plot_numerical_histograms,
    plot_all_boxplots, plot_hist_by_category
)

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Исходные данные",
        "Основная информация",
        "Визуализация пропусков",
        "Описательные статистики",
        "Анализ выбросов",
        "Анализ взаимосвязей"
    ])
    if st.sidebar.button("Вернуть исходные данные"):
        st.session_state['df'] = st.session_state['df_original'].copy()
        st.session_state['missing_filled'] = False
        st.session_state['outliers_filled'] = False
        st.rerun()

    # Проверяем, изменился ли файл (по имени и размеру)
    current_file_id = f"{uploaded.name}_{uploaded.size}"
    previous_file_id = st.session_state.get('current_file_id')

    if 'df' not in st.session_state or current_file_id != previous_file_id:
        # Загружаем новый файл
        df_initial = load_data(uploaded)
        # df_initial = pd.read_csv(r"D:\ПОЛИТЕХ\4 курс\Интеллектуальные системы и технологии\курсовая\eda_application\data\Automobile.csv")
        st.session_state['df'] = df_initial.copy()
        st.session_state['df_original'] = df_initial.copy()
        st.session_state['current_file_id'] = current_file_id
        st.session_state['missing_filled'] = False
        st.session_state['outliers_filled'] = False

    df = st.session_state['df']

    with tab1:
        st.header("Исходные данные")
        st.write("#### Первые строки")
        st.dataframe(df.head())
        st.write("#### Последние строки")
        st.dataframe(df.tail())

    with tab2:
        st.header("Основная информация о датасете")
        info = dataset_info(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Кол-во строк", info["shape"][0])
        col2.metric("Кол-во столбцов", info["shape"][1])

        st.write("#### Типы данных")
        st.json(info["dtypes"])

        # Подсчет количества числовых и категориальных колонок
        numeric_cols = df.select_dtypes(include='number').columns
        categorical_cols = df.select_dtypes(include='object').columns

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Числовых признаков", len(numeric_cols))
        with col2:
            st.metric("Категориальных признаков", len(categorical_cols))

        st.write("#### Пропущенные значения")
        st.json(info["missing_values"])

    with tab3:
        st.header("Визуализация пропусков")
        st.pyplot(plot_missing(df))

        missing_filled = st.session_state.get('missing_filled', False)

        # Сохраняем индексы строк с пропусками при первой загрузке файла
        if 'missing_row_indices' not in st.session_state:
            original_missing_rows = get_missing_rows(st.session_state['df_original'])
            st.session_state['missing_row_indices'] = original_missing_rows.index.tolist()

        # Получаем строки с пропусками (или бывшими пропусками)
        missing_row_indices = st.session_state['missing_row_indices']

        if missing_row_indices:
            # Показываем строки по сохраненным индексам из текущего df
            rows_to_show = df.loc[missing_row_indices]

            # Заголовок таблицы
            if missing_filled:
                st.write("#### Строки после заполнения пропусков")
            else:
                st.write("#### Строки с пропущенными значениями")

            st.dataframe(rows_to_show, use_container_width=True)

        has_numeric_missing = df.select_dtypes(include='number').isna().any().any()
        cat_cols = df.select_dtypes(include='object').columns
        has_categorical_missing = df[cat_cols].isna().any().any() if len(cat_cols) > 0 else False

        if has_numeric_missing or has_categorical_missing:
            st.write("#### Заполнение пропусков")
            st.info("Числовые пропуски заполняются медианой.")

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
                # Нет категориальных пропусков — просто информируем
                st.info("Категориальные пропуски отсутствуют. Будут обработаны только числовые.")

            # === Кнопка заполнения ===
            if st.button("Заполнить пропуски"):
                method_key = 'unknown' if cat_method == "Неизвестно (Unknown)" else 'mode'
                df_updated, filled_info = fill_missing_values(
                    st.session_state['df'],
                    cat_method=method_key,
                    group_col=None  # Не используем группировку
                )
                st.session_state['df'] = df_updated
                st.session_state['missing_filled'] = True
                st.rerun()

        else:
            st.info("Все данные полные — заполнение пропусков не требуется.")

        if missing_filled:
            st.success("Пропуски успешно заполнены!")


    with tab4:
        st.header("Описательные статистики")
        numeric_cols = df.select_dtypes(include='number').columns
        cat_cols = df.select_dtypes(include='object').columns

        st.write("#### Числовые признаки")
        st.dataframe(numerical_stats(df))
        # st.dataframe(categorical_stats(df))

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
                    st.json(info['most_common'])

                    if len(info['least_common']) <= 10:
                        st.write("**Наименее частые значения:**")
                        st.json(info['least_common'])

            # Визуализация категориальных данных
            st.write("#### Визуализация категориальных данных")
            column = st.selectbox("Выберите категориальную колонку для визуализации", cat_cols)
            st.plotly_chart(plot_bar(df, column))
        else:
            st.info("Нет категориальных столбцов")


    with tab5:
        st.header("Анализ выбросов")
        if len(numeric_cols) > 0:
            outliers_summary = get_outliers_summary(df)
            total_outliers = sum(info['outliers_count'] for info in outliers_summary.values())

            st.write("#### Визуализация выбросов")
            st.pyplot(plot_all_boxplots(df))

            # === Определяем outlier_col ЗДЕСЬ, до условия ===
            outlier_col = st.selectbox("Выберите признак для детального анализа выбросов", numeric_cols)

            # Таблица и детали — только если есть выбросы
            if total_outliers > 0:
                st.write("#### Обнаруженные выбросы (метод межквартального размаха IQR)")
                outliers_data = []
                for col, info in outliers_summary.items():
                    outliers_data.append({
                        'Признак': col,
                        'Количество выбросов': info['outliers_count'],
                        'Нижняя граница': round(info['lower_bound'], 3),
                        'Верхняя граница': round(info['upper_bound'], 3)
                    })
                st.dataframe(outliers_data, use_container_width=True)

                # Детальный анализ — используем уже определённый outlier_col
                outlier_info = outliers_summary[outlier_col]

                if outlier_info['outliers_count'] > 0:
                    st.write(f"**Найдено выбросов: {outlier_info['outliers_count']}**")
                    st.write(f"Границы нормальных значений: [{outlier_info['lower_bound']:.3f}, {outlier_info['upper_bound']:.3f}]")
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

            # === Кнопки обработки — теперь outlier_col всегда определён ===
            st.write("#### Обработка выбросов")
            st.info("Метод: замена значений за пределами IQR на границы")

            if st.button(f"Заменить выбросы в '{outlier_col}'"):
                df_clean = fill_outliers_iqr(st.session_state['df'], [outlier_col])
                st.session_state['df'] = df_clean
                st.session_state['outliers_filled'] = True
                st.rerun()

            if st.button("Заменить все выбросы"):
                df_clean = fill_outliers_iqr(st.session_state['df'], numeric_cols.tolist())
                st.session_state['df'] = df_clean
                st.session_state['outliers_filled'] = True
                st.rerun()

        else:
            st.info("Нет числовых столбцов для анализа выбросов")


    with tab6:
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
else:
    # Если файл не загружен, показываем инструкцию
    st.info("Для начала работы загрузите CSV-файл через боковую панель для начала анализа данных.")
