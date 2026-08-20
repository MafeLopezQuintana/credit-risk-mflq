"""
Módulo de entrenamiento y evaluación de modelos supervisados
para predecir Pago_atiempo (riesgo crediticio).
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from ft_engineering import pipeline_completo

# Rutas absolutas, para que funcione sin importar desde dónde se ejecute
CARPETA_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET = os.path.join(CARPETA_ACTUAL, '..', '..', 'Base_de_datos.xlsx')
RUTA_MODELO = os.path.join(CARPETA_ACTUAL, '..', 'modelo_final.pkl')


def preparar_datos(ruta_datos):
    """Carga y limpia los datos, separa en X (features) e y (target)."""
    df = pipeline_completo(ruta_datos)
    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']
    return X, y


def dividir_datos(X, y, test_size=0.2, random_state=42):
    """Divide en conjuntos de entrenamiento y prueba, manteniendo la proporción de clases."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def entrenar_modelos(X_train, y_train):
    """Entrena varios modelos supervisados y los devuelve en un diccionario."""
    modelos = {
        'Regresion Logistica': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric='logloss'
        )
    }

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        print(f"✅ {nombre} entrenado")

    return modelos


from sklearn.metrics import fbeta_score

def evaluar_modelos(modelos, X_test, y_test):
    """
    Evalúa cada modelo con métricas de clasificación, calculadas sobre la clase 0
    (clientes que NO pagan a tiempo), que es la clase de interés para el negocio.
    Se incluye F2-Score, que pondera más el Recall (detectar clientes riesgosos)
    que la Precision, alineado con el costo de negocio de un falso negativo.
    """
    resultados = []

    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        metricas = {
            'Modelo': nombre,
            'Precision_clase0': precision_score(y_test, y_pred, pos_label=0),
            'Recall_clase0': recall_score(y_test, y_pred, pos_label=0),
            'F1-Score_clase0': f1_score(y_test, y_pred, pos_label=0),
            'F2-Score_clase0': fbeta_score(y_test, y_pred, beta=2, pos_label=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }
        resultados.append(metricas)

        print(f"\n=== {nombre} ===")
        print(classification_report(y_test, y_pred))
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

    return pd.DataFrame(resultados).sort_values(by='F2-Score_clase0', ascending=False)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def seleccionar_mejor_modelo(modelos, tabla_resultados):
    """Selecciona el modelo con mejor F2-Score en la clase 0 (prioriza Recall: detectar clientes riesgosos)."""
    mejor_nombre = tabla_resultados.iloc[0]['Modelo']
    mejor_modelo = modelos[mejor_nombre]
    print(f"\n🏆 Mejor modelo (prioriza detectar clientes riesgosos): {mejor_nombre}")
    return mejor_nombre, mejor_modelo


def guardar_modelo(modelo, ruta=RUTA_MODELO):
    """Guarda el modelo entrenado en disco usando joblib."""
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en: {ruta}")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

if __name__ == "__main__":
    X, y = preparar_datos(RUTA_DATASET)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    modelos = entrenar_modelos(X_train, y_train)
    tabla_resultados = evaluar_modelos(modelos, X_test, y_test)

    print("\n=== Comparación de modelos ===")
    print(tabla_resultados)

    nombre_mejor, mejor_modelo = seleccionar_mejor_modelo(modelos, tabla_resultados)
    guardar_modelo(mejor_modelo)