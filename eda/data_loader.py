import pandas as pd
import streamlit as st

def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None

