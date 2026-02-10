import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.express as px


# =========================================
# CONFIGURACIÓN GENERAL
# =========================================

st.set_page_config(page_title="EDA Dashboard", layout="wide")

st.title("📊 Dashboard EDA Interactivo")
st.markdown("### Módulo 1: Ingesta y Procesamiento (ETL)")

# =========================================
# SIDEBAR - FUENTE DE DATOS
# =========================================

st.sidebar.header("⚙️ Configuración")

data_source = st.sidebar.radio(
    "Seleccione la fuente de datos:",
    ["Subir CSV", "Subir JSON", "Cargar desde URL"]
)

# =========================================
# FUNCIONES
# =========================================


def convert_financial_columns(df):
    financial_cols = [
        "INGRESOS OPERACIONALES",
        "GANANCIA (PÉRDIDA)",
        "TOTAL ACTIVOS",
        "TOTAL PASIVOS",
        "TOTAL PATRIMONIO"
    ]
    
    for col in financial_cols:
        if col in df.columns:
            
            # Convertir a string por seguridad
            df[col] = df[col].astype(str)
            
            # Eliminar cualquier cosa que no sea número o signo negativo
            df[col] = df[col].str.replace(r"[^\d\-]", "", regex=True)
            
            # Convertir a numérico
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Convertir año
    if "Año de Corte" in df.columns:
        df["Año de Corte"] = pd.to_numeric(
            df["Año de Corte"].astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce"
        )

    st.write("Valores nulos después de conversión:")
    st.dataframe(st.session_state.clean_df[financial_cols].isna().sum())
    
    return df



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
    return None

def detect_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_info = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        mask = (df[col] < lower) | (df[col] > upper)
        outlier_info[col] = {
            "lower": lower,
            "upper": upper,
            "count": mask.sum()
        }

    return outlier_info

def treat_outliers(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower, upper)

    return df

def impute_data(df, method):
    numeric_cols = df.select_dtypes(include=np.number).columns

    if method == "Media":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == "Mediana":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == "Cero":
        df[numeric_cols] = df[numeric_cols].fillna(0)

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
# PROCESAMIENTO
# =========================================

if df is not None:

    if "clean_df" not in st.session_state:
        st.session_state.clean_df = df.copy()

    st.success("Datos cargados correctamente ✔️")

    st.subheader("Vista previa")
    st.dataframe(st.session_state.clean_df.head())

    st.markdown("### Normalización de Tipos de Datos")

    if st.checkbox("Convertir columnas financieras a formato numérico"):
        if st.button("Aplicar conversión de tipos"):
            st.session_state.clean_df = convert_financial_columns(
                st.session_state.clean_df
            )
            st.success("Columnas financieras convertidas correctamente.")
            
            st.write("Tipos de datos actuales:")
            st.dataframe(st.session_state.clean_df.dtypes)
    
    st.markdown("---")
    st.subheader("🧹 Limpieza de Datos")

    # =====================================
    # ELIMINAR DUPLICADOS
    # =====================================

    if st.checkbox("Activar eliminación de duplicados"):

        if st.button("Confirmar eliminación"):
            before = st.session_state.clean_df.shape[0]
            st.session_state.clean_df = st.session_state.clean_df.drop_duplicates()
            after = st.session_state.clean_df.shape[0]
            st.success(f"Se eliminaron {before - after} duplicados.")

    # =====================================
    # IMPUTACIÓN
    # =====================================

    st.markdown("### Imputación de valores nulos")

    imputation_method = st.selectbox(
        "Seleccione método:",
        ["Ninguno", "Media", "Mediana", "Cero"]
    )

    if imputation_method != "Ninguno":
        if st.button("Aplicar imputación"):
            st.session_state.clean_df = impute_data(
                st.session_state.clean_df,
                imputation_method
            )
            st.success(f"Imputación aplicada con método: {imputation_method}")

    # =====================================
    # OUTLIERS
    # =====================================

    st.markdown("### Detección de Outliers (IQR)")

    if st.checkbox("Detectar outliers"):

        outlier_info = detect_outliers_iqr(st.session_state.clean_df)

        st.write("Cantidad de outliers por variable:")
        outlier_df = pd.DataFrame(outlier_info).T
        st.dataframe(outlier_df[["count"]])

        # Visualización
        numeric_cols = st.session_state.clean_df.select_dtypes(include=np.number).columns

        selected_col = st.selectbox("Selecciona variable para visualizar:", numeric_cols)

        fig = px.box(
            st.session_state.clean_df,
            y=selected_col,
            title=f"Boxplot - {selected_col}",
            points="outliers"  # muestra los outliers resaltados
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title=selected_col,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)


        # Botón para tratar
        if st.button("Tratar outliers (Winsorización)"):
            st.session_state.clean_df = treat_outliers(st.session_state.clean_df)
            st.success("Outliers tratados correctamente.")

    # =====================================
    # DATASET FINAL
    # =====================================

    st.markdown("---")
    st.subheader("📊 Dataset Procesado")
    st.dataframe(st.session_state.clean_df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", st.session_state.clean_df.shape[0])
    col2.metric("Columnas", st.session_state.clean_df.shape[1])
    col3.metric("Valores Nulos", st.session_state.clean_df.isna().sum().sum())

else:
    st.info("Esperando carga de datos...")
