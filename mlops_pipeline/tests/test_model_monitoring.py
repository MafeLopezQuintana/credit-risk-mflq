"""
Tests unitarios para model_monitoring.py
Valida las métricas de drift (KS, PSI, Jensen-Shannon, Chi-cuadrado),
la combinación de resultados, el análisis por periodo y las alertas.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_monitoring import (
    calcular_ks,
    calcular_psi,
    calcular_js_divergence,
    calcular_chi_cuadrado,
    detectar_drift_numericas,
    detectar_drift_categoricas,
    monitoreo_por_periodo,
    generar_alertas,
)


# ---------- calcular_ks ----------

def test_calcular_ks_sin_diferencia_da_pvalue_alto():
    np.random.seed(0)
    referencia = pd.Series(np.random.normal(0, 1, 500))
    nuevo = pd.Series(np.random.normal(0, 1, 500))
    _, p_value = calcular_ks(referencia, nuevo)
    assert p_value > 0.05


def test_calcular_ks_con_diferencia_da_pvalue_bajo():
    np.random.seed(0)
    referencia = pd.Series(np.random.normal(0, 1, 500))
    nuevo = pd.Series(np.random.normal(10, 1, 500))
    _, p_value = calcular_ks(referencia, nuevo)
    assert p_value < 0.05


# ---------- calcular_psi ----------

def test_calcular_psi_distribuciones_iguales_da_valor_bajo():
    np.random.seed(1)
    referencia = pd.Series(np.random.normal(0, 1, 1000))
    nuevo = pd.Series(np.random.normal(0, 1, 1000))
    psi = calcular_psi(referencia, nuevo)
    assert psi < 0.1


def test_calcular_psi_distribuciones_distintas_da_valor_alto():
    np.random.seed(1)
    referencia = pd.Series(np.random.normal(0, 1, 1000))
    nuevo = pd.Series(np.random.normal(5, 1, 1000))
    psi = calcular_psi(referencia, nuevo)
    assert psi > 0.25


# ---------- calcular_js_divergence ----------

def test_js_divergence_identicas_es_cero():
    referencia = pd.Series([1, 2, 3, 4, 5] * 20)
    nuevo = pd.Series([1, 2, 3, 4, 5] * 20)
    js = calcular_js_divergence(referencia, nuevo)
    assert js == pytest.approx(0.0, abs=0.05)


def test_js_divergence_distintas_es_mayor_a_cero():
    referencia = pd.Series(np.random.normal(0, 1, 500))
    nuevo = pd.Series(np.random.normal(8, 1, 500))
    js = calcular_js_divergence(referencia, nuevo)
    assert js > 0.1


# ---------- calcular_chi_cuadrado ----------

def test_chi_cuadrado_misma_proporcion_da_pvalue_alto():
    referencia = pd.Series(['A', 'B'] * 100)
    nuevo = pd.Series(['A', 'B'] * 100)
    _, p_value = calcular_chi_cuadrado(referencia, nuevo)
    assert p_value > 0.05


def test_chi_cuadrado_proporcion_distinta_da_pvalue_bajo():
    referencia = pd.Series(['A'] * 190 + ['B'] * 10)
    nuevo = pd.Series(['A'] * 10 + ['B'] * 190)
    _, p_value = calcular_chi_cuadrado(referencia, nuevo)
    assert p_value < 0.05


# ---------- detectar_drift_numericas ----------

def test_detectar_drift_numericas_sin_drift():
    np.random.seed(2)
    df_ref = pd.DataFrame({'col1': np.random.normal(0, 1, 300)})
    df_nuevo = pd.DataFrame({'col1': np.random.normal(0, 1, 300)})
    resultado = detectar_drift_numericas(df_ref, df_nuevo, ['col1'])
    assert resultado.loc[0, 'drift_detectado'] == False


def test_detectar_drift_numericas_con_drift():
    np.random.seed(2)
    df_ref = pd.DataFrame({'col1': np.random.normal(0, 1, 300)})
    df_nuevo = pd.DataFrame({'col1': np.random.normal(10, 1, 300)})
    resultado = detectar_drift_numericas(df_ref, df_nuevo, ['col1'])
    assert resultado.loc[0, 'drift_detectado'] == True


def test_detectar_drift_numericas_ignora_columnas_inexistentes():
    df_ref = pd.DataFrame({'col1': [1, 2, 3]})
    df_nuevo = pd.DataFrame({'col1': [1, 2, 3]})
    resultado = detectar_drift_numericas(df_ref, df_nuevo, ['col1', 'columna_que_no_existe'])
    assert len(resultado) == 1


# ---------- detectar_drift_categoricas ----------

def test_detectar_drift_categoricas_sin_drift():
    df_ref = pd.DataFrame({'categoria': ['A', 'B'] * 100})
    df_nuevo = pd.DataFrame({'categoria': ['A', 'B'] * 100})
    resultado = detectar_drift_categoricas(df_ref, df_nuevo, ['categoria'])
    assert resultado.loc[0, 'drift_detectado'] == False


def test_detectar_drift_categoricas_con_drift():
    df_ref = pd.DataFrame({'categoria': ['A'] * 190 + ['B'] * 10})
    df_nuevo = pd.DataFrame({'categoria': ['A'] * 10 + ['B'] * 190})
    resultado = detectar_drift_categoricas(df_ref, df_nuevo, ['categoria'])
    assert resultado.loc[0, 'drift_detectado'] == True


# ---------- monitoreo_por_periodo ----------

def test_monitoreo_por_periodo_devuelve_columnas_esperadas():
    fechas = pd.date_range('2024-01-01', periods=200, freq='D')
    np.random.seed(3)
    tabla = pd.DataFrame({
        'fecha_prestamo': fechas,
        'monto': np.random.normal(0, 1, 200),
    })
    resultado = monitoreo_por_periodo(tabla, columnas_numericas=['monto'], periodo='ME')
    assert set(['periodo', 'columna', 'ks_pvalue', 'psi', 'drift_detectado']).issubset(resultado.columns)


def test_monitoreo_por_periodo_con_pocos_datos_devuelve_vacio():
    tabla = pd.DataFrame({
        'fecha_prestamo': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'monto': [1, 2],
    })
    resultado = monitoreo_por_periodo(tabla, columnas_numericas=['monto'], periodo='ME')
    assert len(resultado) == 0


# ---------- generar_alertas ----------

def test_generar_alertas_sin_drift():
    tabla = pd.DataFrame({'drift_detectado': [False, False, False]})
    alertas = generar_alertas(tabla)
    assert alertas['nivel'] == "NINGUNA"
    assert alertas['alerta_critica'] == False


def test_generar_alertas_moderada():
    tabla = pd.DataFrame({'drift_detectado': [True, False, False, False, False]})
    alertas = generar_alertas(tabla, umbral_proporcion=0.3)
    assert alertas['nivel'] == "MODERADA"
    assert alertas['alerta_critica'] == False


def test_generar_alertas_critica():
    tabla = pd.DataFrame({'drift_detectado': [True, True, True, False, False]})
    alertas = generar_alertas(tabla, umbral_proporcion=0.3)
    assert alertas['nivel'] == "CRÍTICA"
    assert alertas['alerta_critica'] == True
    assert "reentrenamiento" in alertas['recomendacion'].lower()


def test_generar_alertas_tabla_vacia():
    tabla = pd.DataFrame({'drift_detectado': []})
    alertas = generar_alertas(tabla)
    assert alertas['total_columnas'] == 0
    assert alertas['alerta_critica'] == False