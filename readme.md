# 🏦 Modelo de Riesgo Crediticio - Proyecto Integrador PI5

Proyecto de Data Science desarrollado como parte del rol de Científico de Datos Junior Advanced en el equipo de Datos y Analítica de una empresa financiera. El objetivo es predecir el comportamiento de pago de nuevos clientes de crédito, utilizando información histórica, para anticipar el riesgo de no pago.

## 📋 Caso de negocio

La empresa necesita anticipar qué clientes tienen mayor probabilidad de **no pagar a tiempo** sus créditos, para poder tomar decisiones informadas al momento de otorgar nuevos préstamos. Se cuenta con un histórico de 10,763 créditos otorgados, con información sociodemográfica, financiera y de comportamiento crediticio de cada cliente.

**Variable objetivo:** `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo)

## 🗂️ Estructura del proyecto


## ⚙️ Instalación y configuración

```bash
# Clonar el repositorio
git clone https://github.com/MafeLopezQuintana/credit-risk-mflq.git
cd credit-risk-mflq

# Crear y activar entorno virtual
python -m venv creditrisk-venv
creditrisk-venv\Scripts\activate      # Windows
# source creditrisk-venv/bin/activate # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt
```

## 🔍 1. Análisis Exploratorio de Datos (EDA)

### Calidad de datos detectada
- Dataset de 10,763 registros y 23 columnas, sin filas duplicadas.
- Nulos relevantes en `promedio_ingresos_datacredito` y `tendencia_ingresos` (~27% cada una). Nulos menores en `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor` y `puntaje_datacredito`.
- `tendencia_ingresos` contenía ~58 registros con valores numéricos inválidos en lugar de categorías esperadas (Creciente/Decreciente/Estable).
- `edad_cliente` presentaba valores imposibles (hasta 123 años).
- `puntaje` y `puntaje_datacredito` presentaban valores negativos, inválidos para un score.
- `salario_cliente` y `total_otros_prestamos` presentaban outliers extremos (hasta miles de millones), posibles errores de digitación.

### Variable objetivo desbalanceada
`Pago_atiempo` está fuertemente desbalanceada: **~95% clase 1** (pagó a tiempo) vs. **~5% clase 0** (no pagó a tiempo). Este desbalance se tuvo en cuenta durante el entrenamiento y la evaluación de los modelos.

### Hallazgo crítico: sospecha de data leakage
Se detectó una correlación extremadamente alta (**0.92**) entre la variable `puntaje` y el target `Pago_atiempo`, con una separación casi perfecta entre clases en el análisis multivariable (pairplot). Esto es sospechoso de **data leakage** (posible cálculo del score usando información posterior al otorgamiento del crédito). Por precaución, esta variable fue **excluida del modelado**.

### Multicolinealidad
Se detectó correlación alta entre algunas variables predictoras: `capital_prestado`-`cuota_pactada` (0.76), `saldo_total`-`saldo_principal` (0.73), `cant_creditosvigentes`-`creditos_sectorFinanciero` (0.79).

## 🛠️ 2. Ingeniería de características

El pipeline de limpieza (`ft_engineering.py`) aplica, en orden:
1. Filtrado de edades fuera de rango (18-90 años), imputando con la mediana.
2. Corrección de valores negativos en `puntaje` y `puntaje_datacredito`, imputando con la mediana.
3. Limpieza de valores inválidos en `tendencia_ingresos`, imputando con la moda.
4. Recorte de outliers extremos en `salario_cliente` y `total_otros_prestamos` (percentil 99).
5. Imputación de nulos en variables de saldo con 0 (asumiendo ausencia de producto/saldo).
6. Imputación de `promedio_ingresos_datacredito` con la mediana.
7. Eliminación de `puntaje` (sospecha de data leakage) y `fecha_prestamo` (no utilizada en el modelo).
8. Codificación One-Hot de variables categóricas.

Resultado: dataset limpio de 10,763 filas, 22 columnas, **0 nulos**.

## 🤖 3. Modelado

Se entrenaron y compararon 3 modelos supervisados de clasificación binaria: **Regresión Logística**, **Random Forest** y **XGBoost**, todos configurados para compensar el desbalance de clases (`class_weight='balanced'` / `scale_pos_weight`).

### Hallazgo importante sobre la métrica de evaluación

Al evaluar inicialmente con F1-Score sobre la clase mayoritaria (1 = pagó), Random Forest aparentaba ser el mejor modelo (F1 = 0.976). Sin embargo, su **Recall sobre la clase 0 (no pagó) era de apenas 0.04** — es decir, detectaba solo 4 de 102 clientes riesgosos, resultando inútil para el objetivo de negocio.

Se recalcularon las métricas enfocadas en la clase 0 (clientes riesgosos), y se utilizó **F2-Score** (que pondera más el Recall que la Precision), dado que en riesgo crediticio un falso negativo (no detectar a un cliente que no pagará) es más costoso que un falso positivo.

### Resultados finales (métricas sobre clase 0 - clientes riesgosos)

| Modelo | Precision | Recall | F1-Score | F2-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Regresión Logística** ✅ | 0.070 | **0.569** | 0.125 | **0.235** | 0.625 |
| XGBoost | 0.124 | 0.176 | 0.146 | 0.163 | 0.632 |
| Random Forest | 0.500 | 0.039 | 0.073 | 0.048 | 0.635 |

**Modelo seleccionado: Regresión Logística**, por ser el que mejor detecta clientes riesgosos (Recall = 0.57), priorizando la capacidad de anticipar el no pago sobre la cantidad de falsas alarmas.

### Limitación reconocida
El ROC-AUC de los 3 modelos ronda 0.62-0.64, sugiriendo un techo de performance moderado con las variables disponibles (excluyendo `puntaje` por sospecha de leakage). Existe margen de mejora explorando nuevas variables o validando el origen real de `puntaje`.

## 📈 4. Monitoreo y Data Drift

Se implementó una función de detección de drift (`model_monitoring.py`) basada en el **test de Kolmogorov-Smirnov**, que compara la distribución de cada variable numérica entre un dataset de referencia y uno nuevo.

**Validación de la metodología:**
- Partición **aleatoria** del dataset (control): **0 de 18** columnas con drift detectado ✅ (confirma que la función no genera falsos positivos).
- Partición **secuencial/temporal** (créditos antiguos vs. recientes): **12 de 18** columnas con drift detectado, destacando `edad_cliente` (KS=0.64) y `puntaje_datacredito` (KS=0.24).

**Conclusión:** el perfil de los clientes atendidos ha cambiado a lo largo del tiempo cubierto por el dataset. Se recomienda monitorear este comportamiento en producción y reentrenar el modelo periódicamente para evitar degradación de performance.

## 🖥️ 5. Aplicación de predicción (Streamlit)

Se desarrolló una interfaz visual (`app_streamlit.py`) donde se pueden ingresar los datos de un cliente nuevo y obtener la predicción del modelo en tiempo real.

```bash
cd mlops_pipeline/src
streamlit run app_streamlit.py
```

**Hallazgo de comportamiento del modelo:** pruebas manuales mostraron que `saldo_mora` es la variable con mayor influencia en la predicción — su sola presencia lleva la probabilidad de pago a valores cercanos a 0%, casi sin importar el resto de las variables. Además, dado que el modelo prioriza Recall sobre Precision (por diseño), tiende a generar una proporción alta de falsas alarmas (Precision de clase 0: 0.07). **Se recomienda usar el modelo como señal de alerta para revisión manual, no como filtro de rechazo automático.**

## 🔄 Flujo de trabajo (Git)

El proyecto sigue un flujo de ramas estructurado:
- `developer`: desarrollo activo
- `certification`: validación antes de producción
- `main`: versión oficial, con tags de versión (V1.0.0, V1.0.1, V1.1.0, V1.1.1, V1.2.0)

## 👤 Autora

María Fernanda López Quintana - Científica de Datos Junior Advanced