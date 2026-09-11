"""
Tests adicionales para model_monitoring.py que ejercitan las funciones
que dependen de los archivos reales del proyecto (dataset y modelo
entrenado), para subir la cobertura de las funciones de integración.
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_monitoring import (
    RUTA_DATASET,
    RUTA_MODELO,
    generar_tabla_predicciones,
    generar_reporte_drift,
    generar_reporte_drift_crudo,
)


# ---------- generar_tabla_predicciones ----------

def test_generar_tabla_predicciones_incluye_prediccion_y_probabilidad():
    tabla = generar_tabla_predicciones(RUTA_DATASET, RUTA_MODELO)
    assert 'prediccion' in tabla.columns
    assert 'probabilidad_pago' in tabla.columns


def test_generar_tabla_predicciones_probabilidad_entre_0_y_1():
    tabla = generar_tabla_predicciones(RUTA_DATASET, RUTA_MODELO)
    assert tabla['probabilidad_pago'].between(0, 1).all()


def test_generar_tabla_predicciones_predicciones_son_0_o_1():
    tabla = generar_tabla_predicciones(RUTA_DATASET, RUTA_MODELO)
    assert set(tabla['prediccion'].unique()).issubset({0, 1})


def test_generar_tabla_predicciones_incluye_fecha_si_esta_disponible():
    tabla = generar_tabla_predicciones(RUTA_DATASET, RUTA_MODELO, incluir_fecha=True)
    assert 'fecha_prestamo' in tabla.columns


def test_generar_tabla_predicciones_sin_fecha_si_se_pide():
    tabla = generar_tabla_predicciones(RUTA_DATASET, RUTA_MODELO, incluir_fecha=False)
    assert 'fecha_prestamo' not in tabla.columns


# ---------- generar_reporte_drift ----------

def test_generar_reporte_drift_devuelve_columnas_esperadas():
    reporte = generar_reporte_drift(RUTA_DATASET, RUTA_DATASET)
    columnas_esperadas = {'columna', 'tipo', 'ks_estadistico', 'ks_pvalue', 'psi', 'js_divergence', 'drift_detectado'}
    assert columnas_esperadas.issubset(reporte.columns)


def test_generar_reporte_drift_comparando_dataset_consigo_mismo_no_da_drift():
    reporte = generar_reporte_drift(RUTA_DATASET, RUTA_DATASET)
    assert reporte['drift_detectado'].sum() == 0


# ---------- generar_reporte_drift_crudo ----------

def test_generar_reporte_drift_crudo_incluye_numericas_y_categoricas():
    reporte = generar_reporte_drift_crudo(RUTA_DATASET, RUTA_DATASET)
    tipos = set(reporte['tipo'].unique())
    assert 'numerica' in tipos
    assert 'categorica' in tipos


def test_generar_reporte_drift_crudo_sin_drift_al_comparar_consigo_mismo():
    reporte = generar_reporte_drift_crudo(RUTA_DATASET, RUTA_DATASET)
    assert reporte['drift_detectado'].sum() == 0