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


def create_financial_features(df):

    if (
        "GANANCIA (PÉRDIDA)" in df.columns and
        "INGRESOS OPERACIONALES" in df.columns
    ):
        df["MARGEN_NETO"] = (
            df["GANANCIA (PÉRDIDA)"] /
            df["INGRESOS OPERACIONALES"]
        )

    if (
        "TOTAL PASIVOS" in df.columns and
        "TOTAL ACTIVOS" in df.columns
    ):
        df["RATIO_ENDEUDAMIENTO"] = (
            df["TOTAL PASIVOS"] /
            df["TOTAL ACTIVOS"]
        )

    if (
        "GANANCIA (PÉRDIDA)" in df.columns and
        "TOTAL ACTIVOS" in df.columns
    ):
        df["ROA"] = (
            df["GANANCIA (PÉRDIDA)"] /
            df["TOTAL ACTIVOS"]
        )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


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

def detect_outliers_iqr(df, multiplier=1.5):
    numeric_cols = df.select_dtypes(include=np.number).columns
    outlier_info = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        mask = (df[col] < lower) | (df[col] > upper)

        outlier_info[col] = {
            "lower": lower,
            "upper": upper,
            "count": int(mask.sum())
        }

    return outlier_info


def treat_outliers(df, multiplier=1.5):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        df[col] = np.clip(df[col], lower, upper)

    return df

def apply_log_transform(df, columns):
    for col in columns:
        if col in df.columns:
            df[f"{col}_LOG"] = np.where(
                df[col] > 0,
                np.log1p(df[col]),
                np.nan
            )
    return df



def impute_data(df, method):

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include="object").columns

    # Imputar numéricas
    if method == "Media":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    elif method == "Mediana":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    elif method == "Cero":
        df[numeric_cols] = df[numeric_cols].fillna(0)

    # Imputar categóricas con moda
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

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

    st.markdown("### Detección y Tratamiento de Outliers (IQR)")

    if st.checkbox("Activar módulo de outliers"):
    
        multiplier = st.slider(
            "Multiplicador IQR (más alto = menos agresivo)",
            min_value=1.5,
            max_value=5.0,
            value=3.0,
            step=0.5
        )
    
        outlier_info = detect_outliers_iqr(
            st.session_state.clean_df,
            multiplier=multiplier
        )
    
        st.write("Cantidad de outliers por variable:")
        outlier_df = pd.DataFrame(outlier_info).T
        st.dataframe(outlier_df[["count"]])
    
        numeric_cols = st.session_state.clean_df.select_dtypes(include=np.number).columns
    
        selected_col = st.selectbox(
            "Selecciona variable para visualizar:",
            numeric_cols
        )
    
        fig = px.box(
            st.session_state.clean_df,
            y=selected_col,
            title=f"Boxplot - {selected_col}",
            points="outliers"
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        action = st.selectbox(
            "Selecciona acción:",
            [
                "Solo detectar",
                "Winsorizar",
                "Eliminar extremos extremos (5x IQR)",
                "Aplicar transformación log"
            ]
        )
    
        if st.button("Aplicar acción"):
    
            if action == "Winsorizar":
                st.session_state.clean_df = treat_outliers(
                    st.session_state.clean_df,
                    multiplier=multiplier
                )
                st.success("Winsorización aplicada correctamente.")
    
            elif action == "Eliminar extremos extremos (5x IQR)":
                info_extreme = detect_outliers_iqr(
                    st.session_state.clean_df,
                    multiplier=5
                )
    
                for col in numeric_cols:
                    lower = info_extreme[col]["lower"]
                    upper = info_extreme[col]["upper"]
                    st.session_state.clean_df = st.session_state.clean_df[
                        (st.session_state.clean_df[col] >= lower) &
                        (st.session_state.clean_df[col] <= upper)
                    ]
    
                st.success("Extremos extremos eliminados.")
    
            elif action == "Aplicar transformación log":
                st.session_state.clean_df = apply_log_transform(
                    st.session_state.clean_df,
                    numeric_cols
                )
                st.success("Transformación log aplicada.")


    # =====================================
    # FEATURE ENGINEERING
    # =====================================
    
    st.markdown("### 🧠 Feature Engineering")
    
    if st.checkbox("Crear indicadores financieros"):
    
        if st.button("Generar nuevas variables"):
            st.session_state.clean_df = create_financial_features(
                st.session_state.clean_df
            )
            st.success("Indicadores financieros creados correctamente.")
    
            st.write("Nuevas columnas agregadas:")
            st.write(
                [
                    col for col in st.session_state.clean_df.columns
                    if col in ["MARGEN_NETO", "RATIO_ENDEUDAMIENTO", "ROA"]
                ]
            )


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
