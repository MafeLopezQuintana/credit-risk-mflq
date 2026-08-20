"""
Módulo de ingeniería de características para el proyecto de riesgo crediticio.
Contiene funciones reutilizables de limpieza, transformación y preparación de datos.
"""

import os
import pandas as pd
import numpy as np

# Ruta absoluta a la carpeta donde vive este script (src/)
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
# Subimos dos niveles para llegar a la raíz del proyecto (src/ -> mlops_pipeline -> raíz)
RUTA_DATASET = os.path.join(CARPETA_ACTUAL, '..', '..', 'Base_de_datos.xlsx')
def cargar_datos(ruta):
    """Carga el dataset desde un archivo Excel."""
    df = pd.read_excel(ruta)
    return df


def limpiar_edad(df):
    """Filtra edades fuera de un rango razonable (18-90 años)."""
    df = df.copy()
    df.loc[(df['edad_cliente'] < 18) | (df['edad_cliente'] > 90), 'edad_cliente'] = np.nan
    df['edad_cliente'] = df['edad_cliente'].fillna(df['edad_cliente'].median())
    return df


def limpiar_puntajes_negativos(df):
    """Reemplaza valores negativos de puntaje y puntaje_datacredito por nulos, luego imputa con la mediana."""
    df = df.copy()
    for col in ['puntaje', 'puntaje_datacredito']:
        df.loc[df[col] < 0, col] = np.nan
        df[col] = df[col].fillna(df[col].median())
    return df


def limpiar_tendencia_ingresos(df):
    """Convierte valores no válidos (numéricos) de tendencia_ingresos a NaN, luego imputa con la moda."""
    df = df.copy()
    categorias_validas = ['Creciente', 'Decreciente', 'Estable']
    df.loc[~df['tendencia_ingresos'].isin(categorias_validas), 'tendencia_ingresos'] = np.nan
    df['tendencia_ingresos'] = df['tendencia_ingresos'].fillna(df['tendencia_ingresos'].mode()[0])
    return df


def limpiar_outliers_salario(df):
    """Recorta valores extremos de salario_cliente y total_otros_prestamos usando el percentil 99."""
    df = df.copy()
    for col in ['salario_cliente', 'total_otros_prestamos']:
        limite_superior = df[col].quantile(0.99)
        df[col] = np.where(df[col] > limite_superior, limite_superior, df[col])
    return df


def imputar_nulos_saldos(df):
    """Imputa con 0 los nulos en columnas de saldo (se asume que nulo = sin saldo/sin producto)."""
    df = df.copy()
    columnas_saldo = ['saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor']
    for col in columnas_saldo:
        df[col] = df[col].fillna(0)
    return df


def imputar_ingresos_datacredito(df):
    """Imputa con la mediana los nulos de promedio_ingresos_datacredito."""
    df = df.copy()
    df['promedio_ingresos_datacredito'] = df['promedio_ingresos_datacredito'].fillna(
        df['promedio_ingresos_datacredito'].median()
    )
    return df


def eliminar_columnas_leakage(df):
    """Elimina columnas sospechosas de data leakage (puntaje interno con correlación anómala al target)."""
    df = df.copy()
    columnas_a_eliminar = ['puntaje']  # sospecha de data leakage detectada en el EDA
    df = df.drop(columns=columnas_a_eliminar)
    return df


def codificar_categoricas(df):
    """Aplica One-Hot Encoding a las columnas categóricas."""
    df = df.copy()
    columnas_categoricas = df.select_dtypes(include=['object', 'str']).columns.tolist()
    df = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)
    return df


def eliminar_columna_fecha(df):
    """Elimina la columna de fecha (no se usa directamente en el modelo por ahora)."""
    df = df.copy()
    if 'fecha_prestamo' in df.columns:
        df = df.drop(columns=['fecha_prestamo'])
    return df


def pipeline_completo(ruta_datos):
    """
    Ejecuta todo el pipeline de limpieza y transformación en orden.
    Devuelve el dataframe listo para el modelado.
    """
    df = cargar_datos(ruta_datos)
    df = limpiar_edad(df)
    df = limpiar_puntajes_negativos(df)
    df = limpiar_tendencia_ingresos(df)
    df = limpiar_outliers_salario(df)
    df = imputar_nulos_saldos(df)
    df = imputar_ingresos_datacredito(df)
    df = eliminar_columnas_leakage(df)
    df = eliminar_columna_fecha(df)
    df = codificar_categoricas(df)
    return df


if __name__ == "__main__":
    df_limpio = pipeline_completo(RUTA_DATASET)   # 👈 antes decía '../Base_de_datos.xlsx'
    print("Shape final:", df_limpio.shape)
    print("Nulos restantes:\n", df_limpio.isnull().sum().sum())
    print(df_limpio.head())
    