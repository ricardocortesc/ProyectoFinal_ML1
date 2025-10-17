"""
Módulo de evaluación de modelos con Prefect
"""

import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from prefect import task
from utils import METRICS_PATH, CLASS_LABELS
import os


@task(name="Predecir")
def predict(model, X_test):
    """
    Realiza predicciones con un modelo
    
    Args:
        model: Modelo entrenado
        X_test: Features de test
    
    Returns:
        tuple: (predicciones, tiempo_inferencia)
    """
    start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start_time
    
    print(f"Predicción completada en {inference_time:.4f} segundos")
    print(f"Predicciones: {len(predictions)} muestras")
    
    return predictions, inference_time


@task(name="Calcular Métricas")
def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de clasificación
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        dict: Diccionario con métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


@task(name="Generar Reporte de Clasificación")
def generate_classification_report(y_true, y_pred):
    """
    Genera reporte detallado de clasificación
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        str: Reporte de clasificación
    """
    report = classification_report(y_true, y_pred, zero_division=0)
    print("\nReporte de Clasificación:")
    print(report)
    return report


@task(name="Calcular Matriz de Confusión")
def calculate_confusion_matrix(y_true, y_pred):
    """
    Calcula matriz de confusión
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
    
    Returns:
        np.array: Matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de Confusión:")
    print(cm)
    return cm


@task(name="Evaluar Modelo")
def evaluate_model(model, X_test, y_test, model_name, substance):
    """
    Evaluación completa de un modelo
    
    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Target de test
        model_name: Nombre del modelo
        substance: Nombre de la sustancia
    
    Returns:
        dict: Resultados de evaluación
    """
    print(f"\n{'='*60}")
    print(f"Evaluando {model_name} para {substance}")
    print(f"{'='*60}")
    
    # Predicción
    y_pred, inference_time = predict(model, X_test)
    
    # Métricas
    metrics = calculate_metrics(y_test, y_pred)
    
    # Reporte
    report = generate_classification_report(y_test, y_pred)
    
    # Matriz de confusión
    cm = calculate_confusion_matrix(y_test, y_pred)
    
    results = {
        'model': model,
        'model_name': model_name,
        'substance': substance,
        'metrics': metrics,
        'report': report,
        'confusion_matrix': cm,
        'inference_time': inference_time,
        'n_samples': len(y_test),
        'y_pred': y_pred
    }
    
    return results


@task(name="Comparar Modelos")
def compare_models(evaluation_results):
    """
    Compara resultados de múltiples modelos
    
    Args:
        evaluation_results: Lista de resultados de evaluación
    
    Returns:
        pd.DataFrame: DataFrame comparativo
    """
    comparison_data = []
    
    for result in evaluation_results:
        row = {
            'Modelo': result['model_name'],
            'Sustancia': result['substance'],
            'Accuracy': result['metrics']['accuracy'],
            'F1 Macro': result['metrics']['f1_macro'],
            'F1 Weighted': result['metrics']['f1_weighted'],
            'Precision Macro': result['metrics']['precision_macro'],
            'Recall Macro': result['metrics']['recall_macro'],
            'Tiempo Inferencia (s)': result['inference_time']
        }
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    return df_comparison


@task(name="Guardar Métricas")
def save_metrics(df_metrics, filename="model_comparison.csv"):
    """
    Guarda métricas en archivo CSV
    
    Args:
        df_metrics: DataFrame con métricas
        filename: Nombre del archivo
    
    Returns:
        str: Ruta del archivo guardado
    """
    os.makedirs(METRICS_PATH, exist_ok=True)
    filepath = os.path.join(METRICS_PATH, filename)
    
    df_metrics.to_csv(filepath, index=False)
    print(f"\nMétricas guardadas en: {filepath}")
    
    return filepath


@task(name="Obtener Mejor Modelo")
def get_best_model(df_comparison, metric='F1 Weighted'):
    """
    Identifica el mejor modelo según una métrica
    
    Args:
        df_comparison: DataFrame comparativo
        metric: Métrica a usar para comparación
    
    Returns:
        dict: Información del mejor modelo
    """
    best_idx = df_comparison[metric].idxmax()
    best_model = df_comparison.iloc[best_idx].to_dict()
    
    print(f"\n{'='*60}")
    print(f"MEJOR MODELO (según {metric}):")
    print(f"{'='*60}")
    for key, value in best_model.items():
        print(f"{key}: {value}")
    
    return best_model


@task(name="Evaluar Todos los Modelos")
def evaluate_all_models(models_dict, X_test, y_test, substance):
    """
    Evalúa todos los modelos entrenados para una sustancia
    
    Args:
        models_dict: Diccionario con modelos entrenados
        X_test: Features de test
        y_test: Target de test
        substance: Nombre de la sustancia
    
    Returns:
        list: Lista con resultados de evaluación
    """
    evaluation_results = []
    
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        result = evaluate_model(model, X_test, y_test, model_name, substance)
        result['training_time'] = model_info['training_time']
        evaluation_results.append(result)
    
    return evaluation_results