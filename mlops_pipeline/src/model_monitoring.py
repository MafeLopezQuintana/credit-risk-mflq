"""
Módulo de monitoreo del modelo: detección de data drift comparando
la distribución de los datos de entrenamiento vs. datos nuevos.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from ft_engineering import pipeline_completo

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET = os.path.join(CARPETA_ACTUAL, '..', '..', 'Base_de_datos.xlsx')


def detectar_drift(df_referencia, df_nuevo, columnas_numericas, umbral_pvalue=0.05):
    """
    Compara la distribución de cada columna numérica entre el dataset de referencia
    (con el que se entrenó el modelo) y un dataset nuevo, usando el test de
    Kolmogorov-Smirnov. Si el p-value es menor al umbral, se considera que hubo drift
    (la distribución cambió de forma estadísticamente significativa).
    """
    resultados = []

    for col in columnas_numericas:
        if col in df_referencia.columns and col in df_nuevo.columns:
            estadistico, p_value = ks_2samp(df_referencia[col].dropna(), df_nuevo[col].dropna())
            hay_drift = p_value < umbral_pvalue

            resultados.append({
                'columna': col,
                'estadistico_ks': estadistico,
                'p_value': p_value,
                'drift_detectado': hay_drift
            })

    tabla = pd.DataFrame(resultados).sort_values(by='p_value')
    return tabla


def generar_reporte_drift(ruta_datos_referencia, ruta_datos_nuevos):
    """
    Genera un reporte comparando el dataset original (referencia) contra
    un dataset nuevo, para detectar drift en las variables numéricas.
    """
    df_ref = pipeline_completo(ruta_datos_referencia)
    df_nuevo = pipeline_completo(ruta_datos_nuevos)

    columnas_numericas = df_ref.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Pago_atiempo' in columnas_numericas:
        columnas_numericas.remove('Pago_atiempo')

    reporte = detectar_drift(df_ref, df_nuevo, columnas_numericas)
    return reporte

if __name__ == "__main__":
    # Prueba con partición ALEATORIA (no por fecha), para validar que la función
    # no detecta drift artificial cuando las muestras son comparables.
    df = pipeline_completo(RUTA_DATASET)
    df_ref = df.sample(frac=0.5, random_state=42)
    df_nuevo = df.drop(df_ref.index)

    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_numericas.remove('Pago_atiempo')

    reporte = detectar_drift(df_ref, df_nuevo, columnas_numericas)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(reporte)

    print(f"\nColumnas con drift detectado: {reporte['drift_detectado'].sum()} de {len(reporte)}")