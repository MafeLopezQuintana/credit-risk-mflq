"""
API de predicción de riesgo crediticio, construida con FastAPI.
Expone el modelo entrenado (modelo_final.pkl) como un servicio web.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA_ACTUAL, '..', 'modelo_final.pkl')

modelo = joblib.load(RUTA_MODELO)

app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Predice si un cliente pagará a tiempo su crédito.",
    version="1.0.0"
)


class DatosCliente(BaseModel):
    """Esquema de entrada: los datos que se necesitan de un cliente nuevo."""
    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tipo_laboral: str
    tendencia_ingresos: str

class LoteClientes(BaseModel):
    """Esquema de entrada para predicción por lotes: una lista de clientes."""
    clientes: List[DatosCliente]
    
def preparar_entrada(datos: DatosCliente) -> pd.DataFrame:
    """Convierte los datos recibidos en el formato que espera el modelo (mismo encoding que en el entrenamiento)."""
    fila = {
        'tipo_credito': datos.tipo_credito,
        'capital_prestado': datos.capital_prestado,
        'plazo_meses': datos.plazo_meses,
        'edad_cliente': datos.edad_cliente,
        'salario_cliente': datos.salario_cliente,
        'total_otros_prestamos': datos.total_otros_prestamos,
        'cuota_pactada': datos.cuota_pactada,
        'puntaje_datacredito': datos.puntaje_datacredito,
        'cant_creditosvigentes': datos.cant_creditosvigentes,
        'huella_consulta': datos.huella_consulta,
        'saldo_mora': datos.saldo_mora,
        'saldo_total': datos.saldo_total,
        'saldo_principal': datos.saldo_principal,
        'saldo_mora_codeudor': datos.saldo_mora_codeudor,
        'creditos_sectorFinanciero': datos.creditos_sectorFinanciero,
        'creditos_sectorCooperativo': datos.creditos_sectorCooperativo,
        'creditos_sectorReal': datos.creditos_sectorReal,
        'promedio_ingresos_datacredito': datos.promedio_ingresos_datacredito,
        'tipo_laboral_Independiente': 1 if datos.tipo_laboral == "Independiente" else 0,
        'tendencia_ingresos_Decreciente': 1 if datos.tendencia_ingresos == "Decreciente" else 0,
        'tendencia_ingresos_Estable': 1 if datos.tendencia_ingresos == "Estable" else 0,
    }
    return pd.DataFrame([fila])


@app.get("/")
def home():
    """Endpoint raíz, para confirmar que la API está viva."""
    return {"mensaje": "API de riesgo crediticio activa. Usá /predecir para hacer una predicción."}


@app.post("/predecir")
def predecir(datos: DatosCliente):
    """Recibe los datos de un cliente y devuelve la predicción del modelo."""
    entrada = preparar_entrada(datos)
    prediccion = int(modelo.predict(entrada)[0])
    probabilidad = float(modelo.predict_proba(entrada)[0][1])

    return {
        "prediccion": prediccion,
        "pago_a_tiempo": bool(prediccion == 1),
        "probabilidad_pago": round(probabilidad, 4)
    }
    
@app.post("/predecir_batch")
def predecir_batch(lote: LoteClientes):
    """Recibe una lista de clientes y devuelve la predicción de cada uno en una sola respuesta."""
    entradas = pd.concat(
        [preparar_entrada(cliente) for cliente in lote.clientes],
        ignore_index=True
    )
    predicciones = modelo.predict(entradas)
    probabilidades = modelo.predict_proba(entradas)[:, 1]

    resultados = []
    for prediccion, probabilidad in zip(predicciones, probabilidades):
        resultados.append({
            "prediccion": int(prediccion),
            "pago_a_tiempo": bool(prediccion == 1),
            "probabilidad_pago": round(float(probabilidad), 4)
        })

    return {
        "resultados": resultados,
        "total_procesados": len(resultados)
    }