import pandas as pd
from groq import Groq
import streamlit as st

def get_ai_insights(df_filtered, api_key):
    """
    Analista virtual que procesa el resumen estadístico y genera recomendaciones.
    """
    if not api_key:
        return "⚠️ Por favor, ingresa tu API Key en la barra lateral."

    try:
        client = Groq(api_key=api_key)
        
        # 1. Obtenemos el resumen estadístico (describe) incluyendo todas las columnas
        stats_desc = df_filtered.describe(include='all').to_string()
        
        # 2. Prompt estructurado según el requerimiento
        prompt = f"""
        Eres un Analista Virtual Senior. Analiza el siguiente resumen estadístico de datos filtrados:
        
        DATOS:
        {stats_desc}
        
        Genera una interpretación en lenguaje natural con estos tres puntos:
        1. TENDENCIAS: Patrones numéricos detectados.
        2. RIESGOS: Anomalías, outliers o problemas de calidad.
        3. OPORTUNIDADES: Acciones de negocio basadas en los datos.

        Reglas: Sé profesional, breve y directo. Idioma: Español.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        return completion.choices[0].message.content

    except Exception as e:
        return f"❌ Error en la IA: {str(e)}"