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


def make_columns_unique(df):
        new_cols = []
        counts = {}
    
        for col in df.columns:
            if col in counts:
                counts[col] += 1
                new_cols.append(f"{col}_{counts[col]}")
            else:
                counts[col] = 0
                new_cols.append(col)
    
        df.columns = new_cols
        return df

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
            new_col = f"{col}_LOG"
            if new_col not in df.columns:   # 👈 evitar duplicados
                df[new_col] = np.where(
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
        st.session_state.clean_df = make_columns_unique(df.copy())

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

    # =====================================================
# MÓDULO 2 - VISUALIZACIÓN DINÁMICA (EDA)
# =====================================================

st.markdown("---")
st.header("📊 Módulo 2: Visualización Dinámica (EDA)")

# Solo una vez y con copia limpia
if "clean_df" in st.session_state:
    df_eda = st.session_state.clean_df.copy()
    # Asegurar unicidad solo si es estrictamente necesario una vez
    df_eda = df_eda.loc[:, ~df_eda.columns.duplicated()] 
else:
    st.stop() # Detener si no hay datos
    
    # 🔎 Verificación defensiva extra
    if df_eda.columns.duplicated().any():
        st.error("Existen columnas duplicadas después de la normalización.")
        st.write(df_eda.columns[df_eda.columns.duplicated()])


    df_eda = make_columns_unique(df_eda)

    # ==============================
    # FILTROS GLOBALES DINÁMICOS
    # ==============================

    st.subheader("🎛️ Filtros Globales")

    col1, col2, col3 = st.columns(3)

    # --------------------------
    # FILTRO CATEGÓRICO
    # --------------------------
    with col1:
        categorical_cols = df_eda.select_dtypes(include="object").columns.tolist()

        selected_category_col = st.selectbox(
            "Columna categórica",
            options=["Ninguna"] + categorical_cols,
            key="cat_filter"
        )

        if selected_category_col != "Ninguna":
            category_values = df_eda[selected_category_col].dropna().unique()
            selected_values = st.multiselect(
                "Valores",
                options=category_values,
                key="cat_values"
            )

            if selected_values:
                df_eda = df_eda[
                    df_eda[selected_category_col].isin(selected_values)
                ]

    # --------------------------
    # FILTRO NUMÉRICO
    # --------------------------
    with col2:
        numeric_cols = df_eda.select_dtypes(include=np.number).columns.tolist()

        selected_numeric_col = st.selectbox(
            "Columna numérica",
            options=["Ninguna"] + numeric_cols,
            key="num_filter"
        )

        if selected_numeric_col != "Ninguna":
            min_val = float(df_eda[selected_numeric_col].min())
            max_val = float(df_eda[selected_numeric_col].max())

            selected_range = st.slider(
                "Rango",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                key="num_range"
            )

            df_eda = df_eda[
                (df_eda[selected_numeric_col] >= selected_range[0]) &
                (df_eda[selected_numeric_col] <= selected_range[1])
            ]

    # --------------------------
    # FILTRO TEMPORAL (si existe)
    # --------------------------
    with col3:
        date_cols = df_eda.select_dtypes(include=["datetime64"]).columns.tolist()

        if date_cols:
            selected_date_col = st.selectbox(
                "Columna fecha",
                date_cols,
                key="date_filter"
            )

            min_date = df_eda[selected_date_col].min()
            max_date = df_eda[selected_date_col].max()

            selected_dates = st.date_input(
                "Rango fechas",
                [min_date, max_date],
                key="date_range"
            )

            if len(selected_dates) == 2:
                df_eda = df_eda[
                    (df_eda[selected_date_col] >= pd.to_datetime(selected_dates[0])) &
                    (df_eda[selected_date_col] <= pd.to_datetime(selected_dates[1]))
                ]

    # =====================================================
    # TABS DE EDA
    # =====================================================

    tab1, tab2, tab3 = st.tabs([
        "📊 Análisis Univariado",
        "🔗 Análisis Bivariado",
        "📈 Evolución Temporal"
    ])

    # =====================================================
    # TAB 1 - UNIVARIADO
    # =====================================================

    with tab1:

        st.subheader("Distribuciones")

        numeric_cols = df_eda.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:

            selected_var = st.selectbox(
                "Variable numérica",
                numeric_cols,
                key="uni_var"
            )

            chart_type = st.radio(
                "Tipo de gráfico",
                ["Histograma", "Boxplot"],
                key="uni_chart"
            )

            if chart_type == "Histograma":
                fig = px.histogram(
                    df_eda,
                    x=selected_var,
                    nbins=30,
                    title=f"Distribución de {selected_var}"
                )
            else:
                fig = px.box(
                    df_eda,
                    y=selected_var,
                    title=f"Boxplot de {selected_var}"
                )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No hay variables numéricas disponibles.")

    # =====================================================
    # TAB 2 - BIVARIADO + CORRELACIÓN
    # =====================================================

    with tab2:
        numeric_df = df_eda.select_dtypes(include=np.number).copy()
        
        # Eliminamos duplicados de columnas si los hubiera por error previo
        numeric_df = numeric_df.loc[:, ~numeric_df.columns.duplicated()]
        
        if len(numeric_df.columns) > 1:
            st.subheader("Matriz de Correlación")
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Heatmap de Correlación")
            st.plotly_chart(fig_corr, use_container_width=True)
    
            st.subheader("Relación entre variables")
            col_x = st.selectbox("Variable X", numeric_df.columns, key="x_var")
            col_y = st.selectbox("Variable Y", numeric_df.columns, key="y_var")
    
            # --- SOLUCIÓN AL ERROR ---
            if col_x == col_y:
                # Si son iguales, creamos un DF con una sola columna y referenciamos el nombre
                df_scatter = numeric_df[[col_x]].dropna()
                fig_scatter = px.scatter(
                    df_scatter, 
                    x=col_x, 
                    y=col_x, # Plotly acepta el mismo nombre de columna aquí
                    trendline="ols"
                )
            else:
                # Si son distintas, pasamos ambas
                df_scatter = numeric_df[[col_x, col_y]].dropna()
                fig_scatter = px.scatter(
                    df_scatter, 
                    x=col_x, 
                    y=col_y, 
                    trendline="ols"
                )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Se necesitan al menos dos variables numéricas.")


    # =====================================================
    # TAB 3 - EVOLUCIÓN TEMPORAL
    # =====================================================

    with tab3:

        date_cols = df_eda.select_dtypes(include=["datetime64"]).columns.tolist()

        if date_cols:

            selected_date_col = st.selectbox(
                "Columna fecha",
                date_cols,
                key="time_date"
            )

            numeric_cols = df_eda.select_dtypes(include=np.number).columns.tolist()

            selected_metric = st.selectbox(
                "Métrica",
                numeric_cols,
                key="time_metric"
            )

            df_time = df_eda.sort_values(selected_date_col)

            fig_line = px.line(
                df_time,
                x=selected_date_col,
                y=selected_metric,
                title=f"Evolución de {selected_metric}"
            )

            st.plotly_chart(fig_line, use_container_width=True)

        else:
            st.info("No existen columnas tipo fecha en el dataset.")



    else:
        st.info("Esperando carga de datos...")
