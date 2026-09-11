"""
Aplicación Streamlit para el proyecto de riesgo crediticio.
Incluye dos secciones: predicción individual de un cliente nuevo,
y un dashboard de monitoreo de data drift del modelo en producción.
"""

import os
import streamlit as st
import pandas as pd
import joblib

from model_monitoring import (
    RUTA_DATASET,
    RUTA_MODELO,
    generar_reporte_drift_crudo,
    generar_tabla_predicciones,
    monitoreo_por_periodo,
    generar_alertas,
)

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))

modelo = joblib.load(RUTA_MODELO)

st.set_page_config(page_title="Riesgo Crediticio", page_icon="🏦", layout="wide")
st.title("🏦 Riesgo Crediticio")

tab_prediccion, tab_monitoreo = st.tabs(["🔮 Predicción", "📊 Monitoreo del modelo"])


# ============================================================
# PESTAÑA 1: Predicción individual (ya existente, sin cambios)
# ============================================================
with tab_prediccion:
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


# ============================================================
# PESTAÑA 2: Monitoreo de data drift (NUEVO)
# ============================================================
with tab_monitoreo:
    st.write(
        "Este panel compara la primera mitad del dataset (referencia) contra la "
        "segunda mitad (datos nuevos) para simular el monitoreo del modelo en producción."
    )

    if st.button("Ejecutar monitoreo"):
        with st.spinner("Calculando métricas de drift..."):
            reporte = generar_reporte_drift_crudo(RUTA_DATASET, RUTA_DATASET)
            tabla_predicciones = generar_tabla_predicciones(RUTA_DATASET)
            alertas = generar_alertas(reporte[reporte['tipo'] == 'numerica'])

        st.session_state['reporte_drift'] = reporte
        st.session_state['tabla_predicciones'] = tabla_predicciones
        st.session_state['alertas'] = alertas

    if 'reporte_drift' in st.session_state:
        reporte = st.session_state['reporte_drift']
        tabla_predicciones = st.session_state['tabla_predicciones']
        alertas = st.session_state['alertas']

        # --- Generación de alertas y recomendaciones ---
        st.subheader("🚨 Alertas y recomendaciones")
        if alertas['alerta_critica']:
            st.error(f"**Alerta {alertas['nivel']}** — {alertas['mensaje']}")
        elif alertas['nivel'] == "MODERADA":
            st.warning(f"**Alerta {alertas['nivel']}** — {alertas['mensaje']}")
        else:
            st.success(f"**Sin alertas** — {alertas['mensaje']}")
        st.info(f"**Recomendación:** {alertas['recomendacion']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Variables analizadas", alertas['total_columnas'])
        col2.metric("Con drift detectado", alertas['columnas_con_drift'])
        col3.metric("% con drift", f"{alertas['proporcion_columnas_con_drift']:.0%}")

        st.divider()

        # --- Tabla de métricas con indicador tipo semáforo ---
        st.subheader("📋 Tabla de métricas de drift por variable")

        def semaforo(row):
            if row['drift_detectado']:
                return "🔴 Alto"
            elif row.get('senales_drift', 0) == 1:
                return "🟡 Moderado"
            return "🟢 Sin cambios"

        reporte_visual = reporte.copy()
        reporte_visual['alerta'] = reporte_visual.apply(semaforo, axis=1)
        st.dataframe(reporte_visual, use_container_width=True)

        st.divider()

        # --- Comparación histórica vs actual para una variable elegida ---
        st.subheader("📈 Distribución histórica vs. actual")
        variables_numericas = reporte[reporte['tipo'] == 'numerica']['columna'].tolist()
        variable_elegida = st.selectbox("Elegí una variable numérica", variables_numericas)

        if variable_elegida:
            mitad = len(tabla_predicciones) // 2
            df_historico = tabla_predicciones.iloc[:mitad]
            df_actual = tabla_predicciones.iloc[mitad:]

            comparacion = pd.DataFrame({
                'Histórico': df_historico[variable_elegida].reset_index(drop=True),
                'Actual': df_actual[variable_elegida].reset_index(drop=True),
            })
            st.bar_chart(comparacion.describe().loc[['mean', 'std', 'min', 'max']])

        st.divider()

        # --- Análisis temporal ---
        st.subheader("🕒 Análisis temporal del drift")
        if 'fecha_prestamo' in tabla_predicciones.columns:
            evolucion = monitoreo_por_periodo(
                tabla_predicciones,
                columnas_numericas=variables_numericas,
                periodo='ME',
            )
            if len(evolucion) > 0:
                psi_promedio_por_periodo = evolucion.groupby('periodo')['psi'].mean()
                st.line_chart(psi_promedio_por_periodo)
                st.caption("PSI promedio de todas las variables por periodo (mensual). Valores más altos indican mayor cambio respecto al primer periodo.")

                periodos_con_drift = evolucion[evolucion['drift_detectado']]['periodo'].nunique()
                if periodos_con_drift > 0:
                    st.warning(f"Se detectaron cambios abruptos en {periodos_con_drift} periodo(s).")
            else:
                st.info("No hay suficientes periodos distintos en los datos para un análisis temporal.")
        else:
            st.info("El dataset no tiene columna de fecha disponible para el análisis temporal.")

        # --- Tabla de datos + predicciones ---
        st.divider()
        st.subheader("🔢 Datos junto con los pronósticos del modelo")
        st.dataframe(
            tabla_predicciones[['prediccion', 'probabilidad_pago']].describe(),
            use_container_width=True,
        )
    else:
        st.info("Hacé clic en 'Ejecutar monitoreo' para calcular las métricas de drift.")