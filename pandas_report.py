import pandas as pd
from pandas_profiling import ProfileReport

# Загрузите ваш датасет
df = pd.read_csv("Automobile.csv")

# Создайте отчёт
profile = ProfileReport(df, title="Анализ датасета в Pandas Profiling", explorative=True)

# Сохраните отчёт в HTML
profile.to_file("automobile_report.html")
