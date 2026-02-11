# final_project_datascience

*Link de streamlit:* https://finalprojectdatascience-aazr858jvxs4dfhyqgbfmr.streamlit.app
*Token IA:* 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

- Juan Diego Llorente
- Sara Valentina Cortes
- Samuel Acosta Aristizabal

## Descripción General
Dashboard interactivo desarrollado en Streamlit para realizar análisis exploratorio de datos (EDA) con capacidades de ETL (Extract, Transform, Load), limpieza de datos y visualización dinámica. La aplicación está diseñada para facilitar el análisis de datos financieros y permite procesar datasets desde múltiples fuentes.

## Características Principales

### Módulo 1: Ingesta y Procesamiento (ETL)

#### 1.1 Fuentes de Datos Soportadas
- **CSV**: Carga de archivos CSV locales
- **JSON**: Importación de archivos JSON estructurados
- **URL**: Descarga directa desde URLs públicas

#### 1.2 Procesamiento de Datos

**Normalización de Columnas**
- Eliminación automática de espacios en blanco
- Gestión de nombres de columnas duplicados mediante sufijos numéricos
- Conversión de columnas financieras a formato numérico

**Limpieza de Datos**
- Eliminación de registros duplicados
- Imputación de valores nulos con tres métodos:
  - Media aritmética
  - Mediana
  - Valor cero

**Detección y Tratamiento de Outliers**
- Método IQR (Interquartile Range) configurable
- Multiplicador ajustable (1.5 - 5.0)
- Visualización mediante boxplots interactivos
- Estrategias de tratamiento:
  - Detección pasiva
  - Winsorización
  - Eliminación de valores extremos (5x IQR)
  - Transformación logarítmica

#### 1.3 Feature Engineering

Generación automática de indicadores financieros:

- **MARGEN_NETO**: Rentabilidad neta sobre ingresos operacionales
```
  MARGEN_NETO = GANANCIA (PÉRDIDA) / INGRESOS OPERACIONALES
```

- **RATIO_ENDEUDAMIENTO**: Proporción de pasivos sobre activos
```
  RATIO_ENDEUDAMIENTO = TOTAL PASIVOS / TOTAL ACTIVOS
```

- **ROA (Return on Assets)**: Retorno sobre activos
```
  ROA = GANANCIA (PÉRDIDA) / TOTAL ACTIVOS
```

### Módulo 2: Visualización Dinámica (EDA)

#### 2.1 Sistema de Filtros Globales

**Filtro Categórico**
- Selección múltiple de valores
- Aplicación dinámica al dataset

**Filtro Numérico**
- Rango deslizante (slider) con límites automáticos
- Filtrado en tiempo real

**Filtro Temporal**
- Selección de rangos de fechas
- Detección automática de columnas tipo datetime

#### 2.2 Análisis Exploratorio

**Análisis Univariado**
- Histogramas de distribución (30 bins configurables)
- Boxplots para detección de outliers
- Selector dinámico de variables numéricas

**Análisis Bivariado**
- Matriz de correlación con heatmap interactivo
- Gráficos de dispersión (scatter plots)
- Análisis de relaciones entre variables

**Evolución Temporal**
- Gráficos de líneas para series temporales
- Ordenamiento automático por fecha
- Visualización de tendencias

## Requisitos Técnicos

### Dependencias
```python
streamlit
pandas
numpy
requests
plotly
```

### Instalación
```bash
pip install streamlit pandas numpy requests plotly
```

## Uso

### Ejecución de la Aplicación
```bash
streamlit run app.py
```

### Flujo de Trabajo Recomendado

1. **Carga de Datos**: Seleccionar fuente (CSV/JSON/URL) y cargar dataset
2. **Conversión de Tipos**: Aplicar normalización a columnas financieras
3. **Limpieza**: 
   - Eliminar duplicados
   - Imputar valores nulos
   - Tratar outliers
4. **Feature Engineering**: Generar indicadores financieros
5. **Análisis Exploratorio**: Utilizar filtros y visualizaciones para insights

## Arquitectura de Datos

### Estado de Sesión

La aplicación utiliza `st.session_state` para mantener persistencia:
```python
st.session_state.clean_df  # DataFrame procesado
```

### Caché de Funciones

Optimización mediante decorador `@st.cache_data`:
- `load_csv()`: Carga de archivos CSV
- `load_json()`: Parseo de JSON
- `load_url()`: Descarga desde URLs

## Funciones Principales

| Función | Propósito |
|---------|-----------|
| `make_columns_unique()` | Gestiona columnas duplicadas |
| `convert_financial_columns()` | Normaliza tipos de datos financieros |
| `detect_outliers_iqr()` | Identifica outliers usando IQR |
| `treat_outliers()` | Aplica winsorización |
| `apply_log_transform()` | Transformación logarítmica |
| `impute_data()` | Imputación de valores faltantes |
| `create_financial_features()` | Generación de ratios financieros |

## Limitaciones y Consideraciones

1. **Valores Infinitos**: La función `create_financial_features()` reemplaza valores infinitos con NaN
2. **Columnas Requeridas**: El feature engineering requiere columnas financieras específicas
3. **Memoria**: Datasets grandes pueden afectar el rendimiento
4. **Tipos de Datos**: La conversión numérica usa coerción con manejo de errores

## Casos de Uso

- Análisis de estados financieros corporativos
- Limpieza y preparación de datasets para modelado
- Exploración visual de datos económicos
- Identificación de patrones y anomalías en datos financieros

## Contribuciones y Extensiones Futuras

Posibles mejoras:
- Exportación de datos procesados
- Modelos de machine learning integrados
- Análisis multivariado avanzado
- Dashboard de KPIs financieros
- Integración con bases de datos

---

**Desarrollado con**: Python 3.x | Streamlit | Plotly Express
