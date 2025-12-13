import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import streamlit as st
from .analysis import (
    dataset_info, numerical_stats, get_categorical_analysis,
    get_outliers_summary
)
from .plots import (
    plot_missing, plot_numerical_histograms, plot_all_boxplots,
    plot_correlation_heatmap, plot_bar
)
from eda.html_report import fig_to_base64

def render_summary_report(df, detect_method='iqr', contamination=0.05):
    import streamlit as st
    from scipy.stats import pearsonr

    html_parts = []
    html_parts.append("<h1>Сводный отчёт по датасету</h1>")

    # Общая информация
    info = dataset_info(df)

    st.markdown("#### Общая информация о датасете")
    st.markdown(f"- Количество строк: **{info['shape'][0]}**")
    st.markdown(f"- Количество столбцов: **{info['shape'][1]}**")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    st.markdown(f"- Числовых признаков: **{len(numeric_cols)}**")
    st.markdown(f"- Категориальных признаков: **{len(categorical_cols)}**")

    missing_total = sum(info['missing_values'].values())
    st.markdown(
        f"- Обнаружено **{missing_total}** пропущенных значений."
        if missing_total > 0 else
        "- Пропущенных значений не обнаружено."
    )

    html_parts.append(f"""
    <h2>Общая информация</h2>
    <ul>
        <li>Строк: <b>{info['shape'][0]}</b></li>
        <li>Столбцов: <b>{info['shape'][1]}</b></li>
        <li>Числовых признаков: <b>{len(numeric_cols)}</b></li>
        <li>Категориальных признаков: <b>{len(categorical_cols)}</b></li>
        <li>Пропущенных значений: <b>{missing_total}</b></li>
    </ul>
    """)

    st.markdown("---")

    # Пропуски
    if missing_total > 0:
        fig = plot_missing(df)
        st.pyplot(fig)

        # Таблица с количеством пропусков
        missing_series = df.isnull().sum().sort_values(ascending=False)
        missing_df = missing_series[missing_series > 0].to_frame(name='Пропущено')
        missing_table_html = missing_df.to_html(classes='table', escape=False)

        html_parts.append(f"""
        <h2>Пропущенные значения</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%; margin-top: 20px;">
        """)

    # Числовые признаки
    if numeric_cols:
        st.markdown("#### Числовые признаки")
        stats = numerical_stats(df)
        st.dataframe(stats)

        html_parts.append("<h2>Числовые признаки</h2>")
        html_parts.append(stats.to_html(border=1))

        fig = plot_numerical_histograms(df)
        st.pyplot(fig)

        html_parts.append(f"""
        <h3>Гистограммы числовых признаков</h3>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%;">
        """)

        st.markdown("---")

    # Категориальные признаки
    if categorical_cols:
        st.markdown("#### Категориальные признаки")

        cat_analysis = get_categorical_analysis(df)

        for col, info_col in cat_analysis.items():
            with st.expander(col):
                st.markdown(f"- Уникальных значений: {info_col['unique_count']}")
                st.markdown(f"- Пропусков: {info_col['missing_count']}")

        # Инициализация в session_state, если ещё не задано
        if "summary_selected_cat_col" not in st.session_state:
            st.session_state["summary_selected_cat_col"] = categorical_cols[0]

        # Selectbox обновляет session_state
        selected_cat_col = st.selectbox(
            "Выберите категориальный признак",
            categorical_cols,
            key="summary_selected_cat_col" 
        )

        fig = plot_bar(df, selected_cat_col)
        st.plotly_chart(fig, key=f"summary_bar_{selected_cat_col}")

        selected_for_html = st.session_state["summary_selected_cat_col"]
        html_parts.append(f"""
        <h2>Категориальные признаки</h2>
        <h3>Распределение по {selected_for_html}</h3>
        {fig.to_html(include_plotlyjs="cdn", full_html=False)}
        """)

        st.markdown("---")

    # Выбросы
    if numeric_cols:
        outliers = get_outliers_summary(df, method=detect_method, contamination=contamination)
        total_outliers = sum(v['outliers_count'] for v in outliers.values())

        if total_outliers > 0:
            st.markdown("#### Анализ выбросов")
            st.markdown(f"Метод: **{detect_method.upper()}**")
            st.markdown(f"Обнаружено **{total_outliers}** выбросов")

            fig = plot_all_boxplots(df)
            st.pyplot(fig)

            html_parts.append(f"""
            <h2>Анализ выбросов</h2>
            <p>Метод: <b>{detect_method.upper()}</b></p>
            <p>Всего выбросов: <b>{total_outliers}</b></p>
            <img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%;">
            """)

            st.markdown("---")

    # Корреляции и гипотезы
    if len(numeric_cols) > 1:
        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)

        html_parts.append(f"""
        <h2>Корреляции</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%;">
        """)

        hypotheses = []

        def safe_corr(a, b):
            mask = df[[a, b]].notna().all(axis=1)
            if mask.sum() < 5:
                return None
            return pearsonr(df.loc[mask, a], df.loc[mask, b])[0]

        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i+1:]:
                r = safe_corr(c1, c2)
                if r and abs(r) >= 0.6:
                    direction = "возрастает" if r > 0 else "убывает"
                    hypotheses.append(f"При увеличении «{c1}» значение «{c2}» {direction} (r={r:.2f})")

        hypotheses = list(dict.fromkeys(hypotheses))

        st.markdown("### Сформулированные гипотезы")
        if hypotheses:
            for h in hypotheses:
                st.markdown(f"- {h}")

            html_parts.append("<h2>Сформулированные гипотезы</h2><ul>")
            for h in hypotheses:
                html_parts.append(f"<li>{h}</li>")
            html_parts.append("</ul>")
        else:
            st.markdown("Сильные корреляции не обнаружены")

    # Сохранение HTML
    st.session_state["summary_html"] = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <title>Сводный EDA-отчёт</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ccc; padding: 6px; }}
        </style>
    </head>
    <body>
        {''.join(html_parts)}
    </body>
    </html>
    """
