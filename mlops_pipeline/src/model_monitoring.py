"""
Módulo de monitoreo del modelo: detección de data drift comparando
la distribución de los datos de entrenamiento vs. datos nuevos, y
seguimiento de las predicciones del modelo a lo largo del tiempo.
"""

import os
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

from ft_engineering import (
    cargar_datos,
    pipeline_completo,
    limpiar_edad,
    limpiar_puntajes_negativos,
    limpiar_tendencia_ingresos,
    limpiar_outliers_salario,
    imputar_nulos_saldos,
    imputar_ingresos_datacredito,
    eliminar_columnas_leakage,
    codificar_categoricas,
)

CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET = os.path.join(CARPETA_ACTUAL, '..', '..', 'Base_de_datos.xlsx')
RUTA_MODELO = os.path.join(CARPETA_ACTUAL, '..', 'modelo_final.pkl')
CARPETA_REPORTES = os.path.join(CARPETA_ACTUAL, '..', '..', 'reports')
RUTA_DRIFT_REPORT = os.path.join(CARPETA_REPORTES, 'drift_report.json')
RUTA_DRIFT_HISTORY = os.path.join(CARPETA_REPORTES, 'drift_history.csv')
UMBRAL_PVALUE_KS = 0.05
UMBRAL_PSI = 0.25          # > 0.25 = cambio significativo (estándar de la industria)
UMBRAL_JS = 0.10           # > 0.10 = cambio moderado/alto
UMBRAL_PVALUE_CHI2 = 0.05
UMBRAL_PROPORCION_ALERTA = 0.30  # si >=30% de las columnas muestran drift, se alerta


# ============================================================
# 1. Métricas de drift individuales
# ============================================================

def calcular_ks(referencia, nuevo):
    """Test de Kolmogorov-Smirnov entre dos muestras numéricas."""
    estadistico, p_value = ks_2samp(referencia.dropna(), nuevo.dropna())
    return estadistico, p_value


def calcular_psi(referencia, nuevo, buckets=10):
    """
    Population Stability Index: mide qué tan distinta es la distribución
    de una variable numérica entre dos periodos, dividiéndola en buckets
    (basados en los percentiles de la referencia) y comparando proporciones.
    Interpretación estándar: <0.1 sin cambio relevante, 0.1-0.25 cambio
    moderado, >0.25 cambio significativo (sugiere revisar/reentrenar).
    """
    referencia = referencia.dropna()
    nuevo = nuevo.dropna()

    puntos_corte = np.percentile(referencia, np.linspace(0, 100, buckets + 1))
    puntos_corte[0] = -np.inf
    puntos_corte[-1] = np.inf
    puntos_corte = np.unique(puntos_corte)

    freq_ref, _ = np.histogram(referencia, bins=puntos_corte)
    freq_nuevo, _ = np.histogram(nuevo, bins=puntos_corte)

    prop_ref = np.where(freq_ref == 0, 1e-6, freq_ref / len(referencia))
    prop_nuevo = np.where(freq_nuevo == 0, 1e-6, freq_nuevo / len(nuevo))

    psi = np.sum((prop_nuevo - prop_ref) * np.log(prop_nuevo / prop_ref))
    return psi


def calcular_js_divergence(referencia, nuevo, bins=10):
    """
    Jensen-Shannon divergence entre las distribuciones de dos muestras
    numéricas (basado en histogramas normalizados). Va de 0 (idénticas)
    a 1 (completamente distintas), usando logaritmo base 2.
    """
    referencia = referencia.dropna()
    nuevo = nuevo.dropna()

    minimo = min(referencia.min(), nuevo.min())
    maximo = max(referencia.max(), nuevo.max())
    limites = np.linspace(minimo, maximo, bins + 1)

    freq_ref, _ = np.histogram(referencia, bins=limites)
    freq_nuevo, _ = np.histogram(nuevo, bins=limites)

    prop_ref = freq_ref / freq_ref.sum() if freq_ref.sum() > 0 else freq_ref
    prop_nuevo = freq_nuevo / freq_nuevo.sum() if freq_nuevo.sum() > 0 else freq_nuevo

    divergencia = jensenshannon(prop_ref, prop_nuevo, base=2)
    return float(divergencia) if not np.isnan(divergencia) else 0.0


def calcular_chi_cuadrado(referencia, nuevo):
    """
    Test de Chi-cuadrado para variables categóricas: compara la
    proporción de cada categoría entre referencia y nuevo dataset.
    """
    categorias = sorted(set(referencia.dropna().unique()) | set(nuevo.dropna().unique()))
    conteo_ref = referencia.value_counts().reindex(categorias, fill_value=0)
    conteo_nuevo = nuevo.value_counts().reindex(categorias, fill_value=0)

    tabla_contingencia = pd.DataFrame({'referencia': conteo_ref, 'nuevo': conteo_nuevo})
    estadistico, p_value, _, _ = chi2_contingency(tabla_contingencia)
    return estadistico, p_value


# ============================================================
# 2. Reporte de drift combinando todas las métricas
# ============================================================

def detectar_drift_numericas(df_referencia, df_nuevo, columnas_numericas):
    """Calcula KS, PSI y Jensen-Shannon para cada columna numérica."""
    resultados = []
    for col in columnas_numericas:
        if col not in df_referencia.columns or col not in df_nuevo.columns:
            continue
        ks_stat, ks_p = calcular_ks(df_referencia[col], df_nuevo[col])
        psi = calcular_psi(df_referencia[col], df_nuevo[col])
        js = calcular_js_divergence(df_referencia[col], df_nuevo[col])

        senales = int(ks_p < UMBRAL_PVALUE_KS) + int(psi > UMBRAL_PSI) + int(js > UMBRAL_JS)

        resultados.append({
            'columna': col,
            'tipo': 'numerica',
            'ks_estadistico': round(ks_stat, 4),
            'ks_pvalue': round(ks_p, 4),
            'psi': round(psi, 4),
            'js_divergence': round(js, 4),
            'senales_drift': senales,
            'drift_detectado': senales >= 2,  # al menos 2 de 3 métricas coinciden
        })
    return pd.DataFrame(resultados)


def detectar_drift_categoricas(df_referencia, df_nuevo, columnas_categoricas):
    """Calcula Chi-cuadrado para cada columna categórica."""
    resultados = []
    for col in columnas_categoricas:
        if col not in df_referencia.columns or col not in df_nuevo.columns:
            continue
        estadistico, p_value = calcular_chi_cuadrado(df_referencia[col], df_nuevo[col])
        resultados.append({
            'columna': col,
            'tipo': 'categorica',
            'chi2_estadistico': round(estadistico, 4),
            'chi2_pvalue': round(p_value, 4),
            'senales_drift': int(p_value < UMBRAL_PVALUE_CHI2),
            'drift_detectado': p_value < UMBRAL_PVALUE_CHI2,
        })
    return pd.DataFrame(resultados)


def generar_reporte_drift(ruta_datos_referencia, ruta_datos_nuevos):
    """
    Genera un reporte combinando drift numérico (KS, PSI, JS) y
    categórico (Chi-cuadrado), comparando dataset de referencia vs nuevo.
    """
    df_ref = pipeline_completo(ruta_datos_referencia)
    df_nuevo = pipeline_completo(ruta_datos_nuevos)

    columnas_numericas = df_ref.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Pago_atiempo' in columnas_numericas:
        columnas_numericas.remove('Pago_atiempo')

    reporte_numerico = detectar_drift_numericas(df_ref, df_nuevo, columnas_numericas)
    return reporte_numerico


def generar_reporte_drift_crudo(ruta_datos_referencia, ruta_datos_nuevos):
    """
    Igual que generar_reporte_drift, pero además calcula Chi-cuadrado sobre
    las variables categóricas originales (antes del One-Hot Encoding),
    ya que el chi-cuadrado necesita las categorías en su forma de texto.
    """
    def preparar_sin_encoding(ruta):
        df = cargar_datos(ruta)
        df = limpiar_edad(df)
        df = limpiar_puntajes_negativos(df)
        df = limpiar_tendencia_ingresos(df)
        df = limpiar_outliers_salario(df)
        df = imputar_nulos_saldos(df)
        df = imputar_ingresos_datacredito(df)
        df = eliminar_columnas_leakage(df)
        return df

    df_ref = preparar_sin_encoding(ruta_datos_referencia)
    df_nuevo = preparar_sin_encoding(ruta_datos_nuevos)

    columnas_numericas = df_ref.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Pago_atiempo' in columnas_numericas:
        columnas_numericas.remove('Pago_atiempo')
    columnas_categoricas = df_ref.select_dtypes(include=['object']).columns.tolist()

    reporte_numerico = detectar_drift_numericas(df_ref, df_nuevo, columnas_numericas)
    reporte_categorico = detectar_drift_categoricas(df_ref, df_nuevo, columnas_categoricas)

    reporte_completo = pd.concat([reporte_numerico, reporte_categorico], ignore_index=True)
    return reporte_completo


# ============================================================
# 3. Tabla de datos + predicciones del modelo
# ============================================================

def generar_tabla_predicciones(ruta_datos, ruta_modelo=RUTA_MODELO, incluir_fecha=True):
    """
    Trae los datos ya procesados junto con la predicción y la probabilidad
    de pago que entrega el modelo para cada registro. Esta es la tabla base
    que se usa tanto para el monitoreo por periodo como para el dashboard.
    """
    modelo = joblib.load(ruta_modelo)

    df_crudo = cargar_datos(ruta_datos)
    df_procesado = pipeline_completo(ruta_datos)

    X = df_procesado.drop(columns=['Pago_atiempo']) if 'Pago_atiempo' in df_procesado.columns else df_procesado
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)[:, 1]

    tabla = df_procesado.copy()
    tabla['prediccion'] = predicciones
    tabla['probabilidad_pago'] = probabilidades

    if incluir_fecha and 'fecha_prestamo' in df_crudo.columns:
        tabla['fecha_prestamo'] = df_crudo['fecha_prestamo'].values

    return tabla


# ============================================================
# 4. Muestreo periódico y análisis temporal
# ============================================================

def monitoreo_por_periodo(tabla_con_fecha, columnas_numericas, columna_fecha='fecha_prestamo', periodo='ME'):
    """
    Divide la tabla en periodos (por defecto mensuales) según columna_fecha,
    toma el primer periodo como referencia, y calcula KS + PSI de cada
    periodo posterior contra esa referencia. Esto da la evolución del
    drift a lo largo del tiempo (muestreo con periodicidad definida).
    """
    tabla = tabla_con_fecha.copy()
    tabla[columna_fecha] = pd.to_datetime(tabla[columna_fecha])
    tabla = tabla.sort_values(columna_fecha)

    grupos = [(nombre, grupo) for nombre, grupo in tabla.groupby(pd.Grouper(key=columna_fecha, freq=periodo)) if len(grupo) > 0]

    if len(grupos) < 2:
        return pd.DataFrame(columns=['periodo', 'columna', 'ks_pvalue', 'psi', 'drift_detectado'])

    nombre_referencia, df_referencia = grupos[0]
    resultados = []

    for nombre_periodo, df_periodo in grupos[1:]:
        for col in columnas_numericas:
            if col not in df_referencia.columns or col not in df_periodo.columns:
                continue
            _, ks_p = calcular_ks(df_referencia[col], df_periodo[col])
            psi = calcular_psi(df_referencia[col], df_periodo[col])
            resultados.append({
                'periodo': nombre_periodo.strftime('%Y-%m'),
                'columna': col,
                'ks_pvalue': round(ks_p, 4),
                'psi': round(psi, 4),
                'drift_detectado': (ks_p < UMBRAL_PVALUE_KS) or (psi > UMBRAL_PSI),
            })

    return pd.DataFrame(resultados)


# ============================================================
# 5. Alertas y recomendaciones automáticas
# ============================================================

def generar_alertas(tabla_drift, umbral_proporcion=UMBRAL_PROPORCION_ALERTA):
    """
    A partir de un reporte de drift (con columna 'drift_detectado'),
    calcula la proporción de variables afectadas y genera un mensaje
    de alerta y una recomendación si se supera el umbral crítico.
    """
    total = len(tabla_drift)
    con_drift = int(tabla_drift['drift_detectado'].sum()) if total > 0 else 0
    proporcion = con_drift / total if total > 0 else 0.0
    alerta_critica = proporcion >= umbral_proporcion

    if alerta_critica:
        nivel = "CRÍTICA"
        mensaje = f"Se detectó drift en {con_drift} de {total} variables ({proporcion:.0%}). Esto supera el umbral de {umbral_proporcion:.0%}."
        recomendacion = "Se recomienda revisar el pipeline de datos y evaluar el reentrenamiento del modelo."
    elif proporcion > 0:
        nivel = "MODERADA"
        mensaje = f"Se detectó drift en {con_drift} de {total} variables ({proporcion:.0%}), por debajo del umbral crítico."
        recomendacion = "Se recomienda seguir monitoreando; no es urgente reentrenar todavía."
    else:
        nivel = "NINGUNA"
        mensaje = "No se detectó drift significativo en las variables analizadas."
        recomendacion = "No se requiere acción."

    return {
        'nivel': nivel,
        'proporcion_columnas_con_drift': round(proporcion, 4),
        'columnas_con_drift': con_drift,
        'total_columnas': total,
        'alerta_critica': alerta_critica,
        'mensaje': mensaje,
        'recomendacion': recomendacion,
    }

# ============================================================
# 6. Exportación de reportes (JSON actual + historial CSV)
# ============================================================

def guardar_drift_report(reporte_df, alertas, ruta_salida=RUTA_DRIFT_REPORT):
    """
    Guarda el reporte de drift de la corrida actual en un archivo JSON,
    con un resumen ejecutivo y el detalle por variable.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    contenido = {
        'timestamp': datetime.now().isoformat(),
        'resumen': {
            'total_columnas': alertas['total_columnas'],
            'columnas_con_drift': alertas['columnas_con_drift'],
            'proporcion_con_drift': alertas['proporcion_columnas_con_drift'],
            'nivel_alerta': alertas['nivel'],
            'mensaje': alertas['mensaje'],
            'recomendacion': alertas['recomendacion'],
        },
        'detalle_por_variable': reporte_df.to_dict(orient='records'),
    }

    with open(ruta_salida, 'w', encoding='utf-8') as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2)

    return ruta_salida


def actualizar_drift_history(reporte_df, ruta_salida=RUTA_DRIFT_HISTORY):
    """
    Agrega la corrida actual como filas nuevas al historial acumulado
    de drift (CSV), para llevar un registro de cómo evoluciona en el tiempo.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    reporte_con_fecha = reporte_df.copy()
    reporte_con_fecha.insert(0, 'fecha_ejecucion', datetime.now().isoformat())

    archivo_existe = os.path.exists(ruta_salida)
    reporte_con_fecha.to_csv(ruta_salida, mode='a', header=not archivo_existe, index=False)

    return ruta_salida

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Prueba con partición ALEATORIA, para validar que la función
    # no detecta drift artificial cuando las muestras son comparables.
    df = pipeline_completo(RUTA_DATASET)
    df_ref = df.sample(frac=0.5, random_state=42)
    df_nuevo = df.drop(df_ref.index)

    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_numericas.remove('Pago_atiempo')

    print("=== Reporte de drift (partición aleatoria, control) ===")
    reporte = detectar_drift_numericas(df_ref, df_nuevo, columnas_numericas)
    print(reporte)

    alertas = generar_alertas(reporte)
    print(f"\nNivel de alerta: {alertas['nivel']}")
    print(f"Mensaje: {alertas['mensaje']}")
    print(f"Recomendación: {alertas['recomendacion']}")

    print("\n=== Tabla de datos + predicciones del modelo ===")
    tabla_predicciones = generar_tabla_predicciones(RUTA_DATASET)
    print(tabla_predicciones[['prediccion', 'probabilidad_pago']].head())

    ruta_json = guardar_drift_report(reporte, alertas)
    ruta_csv = actualizar_drift_history(reporte)
    print(f"\nReporte guardado en: {ruta_json}")
    print(f"Historial actualizado en: {ruta_csv}")
