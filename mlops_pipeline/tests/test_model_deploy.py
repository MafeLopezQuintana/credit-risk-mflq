"""
Tests unitarios para model_deploy.py
Valida la preparación de datos de entrada y los endpoints de la API FastAPI.
"""

import sys
import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_deploy import app, preparar_entrada, DatosCliente

client = TestClient(app)


# Datos de ejemplo válidos, reutilizables en varios tests
def datos_cliente_ejemplo(**overrides):
    base = {
        "tipo_credito": 1,
        "capital_prestado": 5000000.0,
        "plazo_meses": 24,
        "edad_cliente": 35,
        "salario_cliente": 2500000.0,
        "total_otros_prestamos": 0.0,
        "cuota_pactada": 250000.0,
        "puntaje_datacredito": 700.0,
        "cant_creditosvigentes": 1,
        "huella_consulta": 2,
        "saldo_mora": 0.0,
        "saldo_total": 1000000.0,
        "saldo_principal": 900000.0,
        "saldo_mora_codeudor": 0.0,
        "creditos_sectorFinanciero": 1,
        "creditos_sectorCooperativo": 0,
        "creditos_sectorReal": 0,
        "promedio_ingresos_datacredito": 2400000.0,
        "tipo_laboral": "Independiente",
        "tendencia_ingresos": "Estable",
    }
    base.update(overrides)
    return base


# ---------- preparar_entrada ----------

def test_preparar_entrada_devuelve_dataframe():
    datos = DatosCliente(**datos_cliente_ejemplo())
    resultado = preparar_entrada(datos)
    assert isinstance(resultado, pd.DataFrame)
    assert len(resultado) == 1


def test_preparar_entrada_codifica_tipo_laboral_independiente():
    datos = DatosCliente(**datos_cliente_ejemplo(tipo_laboral="Independiente"))
    resultado = preparar_entrada(datos)
    assert resultado.loc[0, 'tipo_laboral_Independiente'] == 1


def test_preparar_entrada_codifica_tipo_laboral_no_independiente():
    datos = DatosCliente(**datos_cliente_ejemplo(tipo_laboral="Empleado"))
    resultado = preparar_entrada(datos)
    assert resultado.loc[0, 'tipo_laboral_Independiente'] == 0


def test_preparar_entrada_codifica_tendencia_decreciente():
    datos = DatosCliente(**datos_cliente_ejemplo(tendencia_ingresos="Decreciente"))
    resultado = preparar_entrada(datos)
    assert resultado.loc[0, 'tendencia_ingresos_Decreciente'] == 1
    assert resultado.loc[0, 'tendencia_ingresos_Estable'] == 0


def test_preparar_entrada_codifica_tendencia_estable():
    datos = DatosCliente(**datos_cliente_ejemplo(tendencia_ingresos="Estable"))
    resultado = preparar_entrada(datos)
    assert resultado.loc[0, 'tendencia_ingresos_Estable'] == 1
    assert resultado.loc[0, 'tendencia_ingresos_Decreciente'] == 0


def test_preparar_entrada_codifica_tendencia_creciente_como_base():
    # "Creciente" es la categoría base: ambas columnas dummy deben quedar en 0
    datos = DatosCliente(**datos_cliente_ejemplo(tendencia_ingresos="Creciente"))
    resultado = preparar_entrada(datos)
    assert resultado.loc[0, 'tendencia_ingresos_Decreciente'] == 0
    assert resultado.loc[0, 'tendencia_ingresos_Estable'] == 0


# ---------- endpoint / ----------

def test_home_responde_ok():
    respuesta = client.get("/")
    assert respuesta.status_code == 200
    assert "mensaje" in respuesta.json()


# ---------- endpoint /predecir ----------

def test_predecir_responde_ok_con_datos_validos():
    respuesta = client.post("/predecir", json=datos_cliente_ejemplo())
    assert respuesta.status_code == 200


def test_predecir_devuelve_estructura_esperada():
    respuesta = client.post("/predecir", json=datos_cliente_ejemplo())
    cuerpo = respuesta.json()
    assert "prediccion" in cuerpo
    assert "pago_a_tiempo" in cuerpo
    assert "probabilidad_pago" in cuerpo


def test_predecir_prediccion_es_0_o_1():
    respuesta = client.post("/predecir", json=datos_cliente_ejemplo())
    cuerpo = respuesta.json()
    assert cuerpo["prediccion"] in [0, 1]


def test_predecir_probabilidad_entre_0_y_1():
    respuesta = client.post("/predecir", json=datos_cliente_ejemplo())
    cuerpo = respuesta.json()
    assert 0.0 <= cuerpo["probabilidad_pago"] <= 1.0


def test_predecir_pago_a_tiempo_coincide_con_prediccion():
    respuesta = client.post("/predecir", json=datos_cliente_ejemplo())
    cuerpo = respuesta.json()
    assert cuerpo["pago_a_tiempo"] == (cuerpo["prediccion"] == 1)


def test_predecir_rechaza_datos_incompletos():
    # Falta el campo 'edad_cliente' -> debe fallar validación (422)
    datos_incompletos = datos_cliente_ejemplo()
    del datos_incompletos["edad_cliente"]
    respuesta = client.post("/predecir", json=datos_incompletos)
    assert respuesta.status_code == 422


def test_predecir_rechaza_tipo_de_dato_invalido():
    # edad_cliente como texto en vez de número -> debe fallar validación
    datos_invalidos = datos_cliente_ejemplo(edad_cliente="treinta y cinco")
    respuesta = client.post("/predecir", json=datos_invalidos)
    assert respuesta.status_code == 422