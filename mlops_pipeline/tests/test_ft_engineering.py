"""
Pruebas unitarias para las funciones de limpieza de ft_engineering.py
"""

import pandas as pd
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ft_engineering import (
    limpiar_edad,
    limpiar_puntajes_negativos,
    limpiar_tendencia_ingresos,
    limpiar_outliers_salario,
    imputar_nulos_saldos,
    eliminar_columnas_leakage,
    eliminar_columna_fecha,
)


def test_limpiar_edad_corrige_valores_fuera_de_rango():
    """Una edad de 150 años (inválida) debe ser reemplazada por la mediana."""
    df = pd.DataFrame({'edad_cliente': [25, 30, 150, 40]})
    resultado = limpiar_edad(df)
    # La edad de 150 ya no debe estar presente
    assert 150 not in resultado['edad_cliente'].values
    # No debe haber nulos después de la limpieza
    assert resultado['edad_cliente'].isnull().sum() == 0


def test_limpiar_edad_mantiene_valores_validos():
    """Las edades dentro del rango válido no deben modificarse."""
    df = pd.DataFrame({'edad_cliente': [25, 30, 45, 60]})
    resultado = limpiar_edad(df)
    assert list(resultado['edad_cliente']) == [25, 30, 45, 60]


def test_limpiar_puntajes_negativos():
    """Los puntajes negativos deben ser reemplazados por la mediana, sin quedar en negativo."""
    df = pd.DataFrame({
        'puntaje': [80, -10, 90, 85],
        'puntaje_datacredito': [700, 750, -5, 780]
    })
    resultado = limpiar_puntajes_negativos(df)
    assert (resultado['puntaje'] >= 0).all()
    assert (resultado['puntaje_datacredito'] >= 0).all()


def test_limpiar_tendencia_ingresos_corrige_valores_invalidos():
    """Valores numéricos inválidos deben convertirse en una de las 3 categorías válidas."""
    df = pd.DataFrame({'tendencia_ingresos': ['Creciente', '8315', 'Estable', '-224714']})
    resultado = limpiar_tendencia_ingresos(df)
    categorias_validas = {'Creciente', 'Decreciente', 'Estable'}
    assert set(resultado['tendencia_ingresos'].unique()).issubset(categorias_validas)


def test_limpiar_outliers_salario_recorta_valores_extremos():
    """Un salario absurdamente alto debe ser recortado al percentil 99."""
    salarios = [3000000] * 99 + [999999999999]  # 99 salarios normales + 1 outlier extremo
    df = pd.DataFrame({
        'salario_cliente': salarios,
        'total_otros_prestamos': [500000] * 100
    })
    resultado = limpiar_outliers_salario(df)
    assert resultado['salario_cliente'].max() < 999999999999


def test_imputar_nulos_saldos_reemplaza_nan_con_cero():
    """Los nulos en columnas de saldo deben imputarse con 0."""
    df = pd.DataFrame({
        'saldo_mora': [100, np.nan, 300],
        'saldo_total': [np.nan, 200, 300],
        'saldo_principal': [100, 200, np.nan],
        'saldo_mora_codeudor': [np.nan, np.nan, 100]
    })
    resultado = imputar_nulos_saldos(df)
    assert resultado.isnull().sum().sum() == 0


def test_eliminar_columnas_leakage_quita_puntaje():
    """La columna 'puntaje' (sospecha de data leakage) debe ser eliminada."""
    df = pd.DataFrame({'puntaje': [80, 90], 'otra_columna': [1, 2]})
    resultado = eliminar_columnas_leakage(df)
    assert 'puntaje' not in resultado.columns
    assert 'otra_columna' in resultado.columns


def test_eliminar_columna_fecha_quita_fecha_prestamo():
    """La columna 'fecha_prestamo' debe ser eliminada si existe."""
    df = pd.DataFrame({'fecha_prestamo': ['2024-01-01'], 'otra_columna': [1]})
    resultado = eliminar_columna_fecha(df)
    assert 'fecha_prestamo' not in resultado.columns


def test_eliminar_columna_fecha_no_falla_si_no_existe():
    """No debe romper si la columna fecha_prestamo ya no está presente."""
    df = pd.DataFrame({'otra_columna': [1, 2]})
    resultado = eliminar_columna_fecha(df)
    assert 'otra_columna' in resultado.columns