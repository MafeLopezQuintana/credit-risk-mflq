"""
Aplicación Streamlit para predecir el riesgo crediticio de un cliente nuevo,
usando el modelo entrenado (modelo_final.pkl).
"""

import os
import streamlit as st
import pandas as pd
import joblib

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA_ACTUAL, '..', 'modelo_final.pkl')

modelo = joblib.load(RUTA_MODELO)

st.title("🏦 Predicción de Riesgo Crediticio")
st.write("Completá los datos del cliente para predecir si pagará a tiempo.")

tipo_credito = st.number_input("Tipo de crédito", min_value=0, value=4)
capital_prestado = st.number_input("Capital prestado", min_value=0.0, value=2000000.0)
plazo_meses = st.number_input("Plazo (meses)", min_value=1, value=12)
edad_cliente = st.number_input("Edad del cliente", min_value=18, max_value=90, value=35)
salario_cliente = st.number_input("Salario del cliente", min_value=0.0, value=3000000.0)
total_otros_prestamos = st.number_input("Total otros préstamos", min_value=0.0, value=500000.0)
cuota_pactada = st.number_input("Cuota pactada", min_value=0.0, value=200000.0)
puntaje_datacredito = st.number_input("Puntaje Datacrédito", min_value=0, max_value=999, value=750)
cant_creditosvigentes = st.number_input("Créditos vigentes", min_value=0, value=2)
huella_consulta = st.number_input("Huella de consulta", min_value=0, value=3)
saldo_mora = st.number_input("Saldo en mora", min_value=0.0, value=0.0)
saldo_total = st.number_input("Saldo total", min_value=0.0, value=15000.0)
saldo_principal = st.number_input("Saldo principal", min_value=0.0, value=14000.0)
saldo_mora_codeudor = st.number_input("Saldo mora codeudor", min_value=0.0, value=0.0)
creditos_sectorFinanciero = st.number_input("Créditos sector financiero", min_value=0, value=2)
creditos_sectorCooperativo = st.number_input("Créditos sector cooperativo", min_value=0, value=0)
creditos_sectorReal = st.number_input("Créditos sector real", min_value=0, value=1)
promedio_ingresos_datacredito = st.number_input("Promedio ingresos Datacrédito", min_value=0.0, value=1200000.0)
tipo_laboral = st.selectbox("Tipo laboral", ["Empleado", "Independiente"])
tendencia_ingresos = st.selectbox("Tendencia de ingresos", ["Creciente", "Estable", "Decreciente"])

if st.button("Predecir"):
    entrada = pd.DataFrame([{
        'tipo_credito': tipo_credito,
        'capital_prestado': capital_prestado,
        'plazo_meses': plazo_meses,
        'edad_cliente': edad_cliente,
        'salario_cliente': salario_cliente,
        'total_otros_prestamos': total_otros_prestamos,
        'cuota_pactada': cuota_pactada,
        'puntaje_datacredito': puntaje_datacredito,
        'cant_creditosvigentes': cant_creditosvigentes,
        'huella_consulta': huella_consulta,
        'saldo_mora': saldo_mora,
        'saldo_total': saldo_total,
        'saldo_principal': saldo_principal,
        'saldo_mora_codeudor': saldo_mora_codeudor,
        'creditos_sectorFinanciero': creditos_sectorFinanciero,
        'creditos_sectorCooperativo': creditos_sectorCooperativo,
        'creditos_sectorReal': creditos_sectorReal,
        'promedio_ingresos_datacredito': promedio_ingresos_datacredito,
        'tipo_laboral_Independiente': 1 if tipo_laboral == "Independiente" else 0,
        'tendencia_ingresos_Decreciente': 1 if tendencia_ingresos == "Decreciente" else 0,
        'tendencia_ingresos_Estable': 1 if tendencia_ingresos == "Estable" else 0,
    }])

    prediccion = modelo.predict(entrada)[0]
    probabilidad = modelo.predict_proba(entrada)[0][1]

    if prediccion == 1:
        st.success(f"✅ Se predice que el cliente PAGARÁ a tiempo (probabilidad: {probabilidad:.2%})")
    else:
        st.error(f"⚠️ Se predice que el cliente NO pagará a tiempo (probabilidad de pago: {probabilidad:.2%})")