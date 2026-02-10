import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

# =========================================
# CONFIGURACIÓN GENERAL
# =========================================

st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("📊 Dashboard EDA Interactivo")
st.markdown("### Módulo 1: Ingesta y Procesamiento (ETL)")

# =========================================
# SIDEBAR - NAVEGACIÓN
# =========================================

st.sidebar.header("⚙️ Configuración")

data_source = st.sidebar.radio(
    "Seleccione la fuente de datos:",
    ["Subir CSV", "Subir JSON", "Cargar desde URL"]
)

# =========================================
# FUNCIONES ETL
# =========================================

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def load_json(file):
    return pd.read_json(file)

@st.cache_data
def load_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        return None

def impute_data(df, method):
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if method == "Media":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == "Mediana":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == "Cero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
    return df

def treat_outliers(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    
    return df

# =========================================
# CARGA DE DATOS
# =========================================

df = None

if data_source == "Subir CSV":
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file:
        df = load_csv(uploaded_file)

elif data_source == "Subir JSON":
    uploaded_file = st.file_uploader("Sube tu archivo JSON", type=["json"])
    if uploaded_file:
        df = load_json(uploaded_file)

elif data_source == "Cargar desde URL":
    url = st.sidebar.text_input("Ingresa la URL del CSV")
    if url:
        df = load_url(url)

# =========================================
# PROCESAMIENTO INTERACTIVO
# =========================================

if df is not None:

    st.success("Datos cargados correctamente ✔️")
    
    st.subheader("Vista previa del dataset")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("🧹 Limpieza de Datos")

    # Eliminar duplicados
    remove_duplicates = st.checkbox("Eliminar duplicados")
    if remove_duplicates:
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        st.info(f"Se eliminaron {before - after} registros duplicados.")

    # Imputación
    imputation_method = st.selectbox(
        "Método de imputación para variables numéricas:",
        ["Ninguno", "Media", "Mediana", "Cero"]
    )

    if imputation_method != "Ninguno":
        df = impute_data(df, imputation_method)
        st.success(f"Imputación aplicada usando: {imputation_method}")

    # Tratamiento de outliers
    treat_outliers_option = st.checkbox("Detectar y tratar outliers (Método IQR)")
    if treat_outliers_option:
        df = treat_outliers(df)
        st.success("Outliers tratados usando método IQR (Winsorización).")

    st.markdown("---")
    st.subheader("📊 Dataset Procesado")
    st.dataframe(df.head())

    st.markdown("### 📈 Información General")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Valores Nulos", df.isna().sum().sum())

else:
    st.info("Esperando carga de datos...")
