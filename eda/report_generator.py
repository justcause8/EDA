import streamlit as st
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import streamlit as st
from .analysis import (
    dataset_info, numerical_stats, get_categorical_analysis,
    get_outliers_summary, generate_insights, get_change_history_df 
)
from .plots import (
    plot_missing, plot_numerical_histograms, plot_all_boxplots,
    plot_correlation_heatmap, plot_bar, plot_hist_by_category
)
from eda.html_report import fig_to_base64

def render_summary_report(df, detect_method='iqr'):
    html_parts = []
    html_parts.append("<h1>Сводный отчёт по датасету</h1>")

    # Инсайты
    st.markdown("#### Ключевые инсайты")
    insights = generate_insights(df)

    for insight in insights:
        st.markdown(f"- {insight}")

    html_parts.append("<h2>Ключевые инсайты</h2><ul>")
    for insight in insights:
        html_parts.append(f"<li>{insight}</li>")
    html_parts.append("</ul>")

    st.markdown("---")

    # Пропущенные значения
    info = dataset_info(df)
    missing_total = sum(info["missing_values"].values())

    # Заголовок выводим всегда
    st.markdown("#### Пропущенные значения")

    if missing_total > 0:
        fig = plot_missing(df)
        st.pyplot(fig)

        html_parts.append(f"""
        <h2>Пропущенные значения</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}"
             style="max-width:100%; margin-top:20px;">
        """)
    else:
        st.success("Пропущенных значений в данных не обнаружено.")
        html_parts.append("<h2>Пропущенные значения</h2><p>Пропущенных значений нет.</p>")
    
    # Таблица истории
    history_missing = st.session_state.get('history_missing', [])
    if history_missing:
        hist_df = get_change_history_df(df, history_missing) 
        
        if hist_df is not None:
            html_table = hist_df.head(100).to_html(classes='table table-striped', index=False, border=0)
            
            html_parts.append(f"""
            <h3>История заполнения пропусков</h3>
            <div style='overflow-x:auto; font-size: 0.9em;'>
                {html_table}
            </div>
            """)

            st.markdown("##### История заполнения пропусков")
            st.dataframe(hist_df, use_container_width=True)
    
    st.markdown("---")
    

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

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

        if "summary_selected_cat_col" not in st.session_state:
            st.session_state["summary_selected_cat_col"] = categorical_cols[0]

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
        outliers = get_outliers_summary(df, method=detect_method)
        total_outliers = sum(v['outliers_count'] for v in outliers.values())

        st.markdown("#### Анализ выбросов")
        fig = plot_all_boxplots(df)
        st.pyplot(fig)
        
        html_parts.append(f"""
        <h2>Анализ выбросов</h2>
        <p>Метод детектирования: <b>{detect_method.upper()}</b></p>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%;">
        """)

        if total_outliers > 0:
            st.markdown(f"Обнаружено **{total_outliers}** выбросов (метод {detect_method.upper()})")
            html_parts.append(f"<p>Текущее кол-во выбросов: <b>{total_outliers}</b></p>")
        else:
            st.success(f"Выбросов не обнаружено (метод {detect_method.upper()}).")
            html_parts.append(f"<p>Выбросов не обнаружено.</p>")
        
        # Таблица истории
        history_outliers = st.session_state.get('history_outliers', [])
        if history_outliers:
            outlier_hist_df = get_change_history_df(df, history_outliers)
            
            if outlier_hist_df is not None:
                html_table = outlier_hist_df.head(100).to_html(classes='table table-striped', index=False, border=0)
                
                html_parts.append(f"""
                <h3>История обработки выбросов</h3>
                <div style='overflow-x:auto; font-size: 0.9em;'>
                    {html_table}
                </div>
                """)
                    
                st.markdown("##### История обработки выбросов")
                st.dataframe(outlier_hist_df, use_container_width=True)

        st.markdown("---")

        # Анализ взаимосвязей
    if len(numeric_cols) > 1:
        st.markdown("#### Анализ взаимосвязей между числовыми признаками")
        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)

        html_parts.append(f"""
        <h2>Анализ взаимосвязей между числовыми признаками</h2>
        <img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%;">
        """)

    # Сравнение распределений по категориям
    if numeric_cols and categorical_cols:
        st.markdown("#### Сравнение распределений по категориям")
        
        # Выбор признаков для сравнения (сохраняем в session_state)
        if "summary_comp_numeric" not in st.session_state:
            st.session_state["summary_comp_numeric"] = numeric_cols[0]
        if "summary_comp_category" not in st.session_state:
            st.session_state["summary_comp_category"] = categorical_cols[0]

        comp_numeric = st.selectbox(
            "Выберите числовой признак",
            numeric_cols,
            key="summary_comp_numeric"
        )
        comp_category = st.selectbox(
            "Выберите категориальный признак",
            categorical_cols,
            key="summary_comp_category"
        )

        fig = plot_hist_by_category(df, comp_numeric, comp_category)
        if fig:
            st.pyplot(fig)
            html_parts.append(f"""
            <h2>Сравнение распределений</h2>
            <p>Числовой признак: <b>{comp_numeric}</b><br>
            Категориальный признак: <b>{comp_category}</b></p>
            <img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%;">
            """)

    # Сформулированные гипотезы
    hypotheses = []
    if len(numeric_cols) > 1:
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
                    hypotheses.append(f"При увеличении {c1} значение {c2} {direction} (r={r:.2f})")

        hypotheses = list(dict.fromkeys(hypotheses))

        st.markdown("### Сформулированные правила")
        if hypotheses:
            for h in hypotheses:
                st.markdown(f"- {h}")
            html_parts.append("<h2>Сформулированные правила</h2><ul>")
            for h in hypotheses:
                html_parts.append(f"<li>{h}</li>")
            html_parts.append("</ul>")
        else:
            st.markdown("Сильные корреляции не обнаружены")
        st.markdown("---")

    st.markdown("### Вывод")
    def is_performance(col):
        return any(kw in col.lower() for kw in ['mpg', 'fuel', 'efficien', 'consumption', 'economy'])
    def is_power(col):
        return any(kw in col.lower() for kw in ['horse', 'power', 'hp', 'engine'])
    def is_size(col):
        return any(kw in col.lower() for kw in ['weight', 'mass', 'displacement', 'engine', 'cylind', 'volume'])
    def is_time(col):
        return any(kw in col.lower() for kw in ['acceleration', 'time', '0-60', '0_60'])
    def is_year(col):
        return any(kw in col.lower() for kw in ['year', 'model', 'date'])

    def has_hypothesis_between(group1_cols, group2_cols):
        for h in hypotheses:
            words = h.lower()
            if any(c.lower() in words for c in group1_cols) and any(c.lower() in words for c in group2_cols):
                return True
        return False

    # Классификация признаков
    performance_cols = [c for c in numeric_cols if is_performance(c)]
    power_cols = [c for c in numeric_cols if is_power(c)]
    size_cols = [c for c in numeric_cols if is_size(c)]
    time_cols = [c for c in numeric_cols if is_time(c)]
    year_cols = [c for c in numeric_cols if is_year(c)]

    conclusion_parts = []

    # 1. Расход vs размеры
    if performance_cols and size_cols and has_hypothesis_between(performance_cols, size_cols):
        perf_name = performance_cols[0]
        size_name = size_cols[0]
        conclusion_parts.append(f"Чем больше {size_name} автомобиля, тем выше его расход топлива (хуже {perf_name}).")

    # 2. Расход vs мощность
    if performance_cols and power_cols and has_hypothesis_between(performance_cols, power_cols):
        perf_name = performance_cols[0]
        power_name = power_cols[0]
        conclusion_parts.append(f"Чем выше {power_name}, тем выше расход топлива (хуже {perf_name}).")

    # 3. Мощность vs размеры двигателя
    if power_cols and size_cols and has_hypothesis_between(power_cols, size_cols):
        power_name = power_cols[0]
        size_name = size_cols[0]
        conclusion_parts.append(f"Автомобили с большим {size_name} обычно имеют большую {power_name}.")

    # 4. Мощность vs ускорение
    if power_cols and time_cols and has_hypothesis_between(power_cols, time_cols):
        power_name = power_cols[0]
        time_name = time_cols[0]
        conclusion_parts.append(f"Более мощные автомобили ({power_name}) разгоняются быстрее (меньше {time_name}).")

    # 5. Изменения по годам
    if year_cols:
        year_name = year_cols[0]
        if has_hypothesis_between(year_cols, performance_cols):
            perf_name = performance_cols[0]
            conclusion_parts.append(f"С течением времени ({year_name}) автомобили стали экономичнее (лучше {perf_name}).")
        if has_hypothesis_between(year_cols, power_cols):
            power_name = power_cols[0]
            conclusion_parts.append(f"С течением времени ({year_name}) мощность двигателей ({power_name}) выросла.")
        if has_hypothesis_between(year_cols, size_cols) and not has_hypothesis_between(year_cols, power_cols):
            size_name = size_cols[0]
            conclusion_parts.append(f"С течением времени ({year_name}) автомобили стали легче и компактнее (меньше {size_name}).")

    # 6. Цена vs мощность
    price_cols = [c for c in numeric_cols if 'price' in c.lower() or 'cost' in c.lower()]
    if price_cols and power_cols and has_hypothesis_between(price_cols, power_cols):
        price_name = price_cols[0]
        power_name = power_cols[0]
        conclusion_parts.append(f"Дорогие автомобили ({price_name}) обычно мощнее ({power_name})")

    # 7. Пробег vs возраст
    mileage_cols = [c for c in numeric_cols if 'mileage' in c.lower() or 'odometer' in c.lower()]
    if mileage_cols and year_cols and has_hypothesis_between(mileage_cols, year_cols):
        mileage_name = mileage_cols[0]
        year_name = year_cols[0]
        conclusion_parts.append(f"Более старые автомобили (меньше {year_name}) чаще имеют больший пробег ({mileage_name})")

    # Формируем итоговый вывод
    if conclusion_parts:
        conclusion_text = "Основные выводы:\n- " + "\n- ".join(conclusion_parts)
    else:
        if hypotheses:
            conclusion_text = "В данных обнаружены статистически значимые зависимости между характеристиками автомобилей"
        else:
            conclusion_text = "Ярко выраженных закономерностей не обнаружено. Возможно, зависимости нелинейные или объём данных недостаточен"

    st.markdown(conclusion_text)
    html_parts.append(f"<h2>Вывод</h2><p>{conclusion_text.replace(chr(10), '<br>')}</p>")

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
