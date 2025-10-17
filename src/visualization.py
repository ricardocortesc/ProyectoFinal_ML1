"""
Módulo de visualización con Prefectsscacasc
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from prefect import task
from utils import FIGURES_PATH, CLASS_LABELS
import os


@task(name="Graficar Matriz de Confusión")
def plot_confusion_matrix(cm, substance, model_name):
    """
    Genera y guarda gráfico de matriz de confusión
    
    Args:
        cm: Matriz de confusión
        substance: Nombre de la sustancia
        model_name: Nombre del modelo
    
    Returns:
        str: Ruta del archivo guardado
    """
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(7), yticklabels=range(7))
    plt.title(f'Matriz de Confusión: {substance} - {model_name}')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.tight_layout()
    
    filename = f"confusion_matrix_{substance}_{model_name}.png"
    filepath = os.path.join(FIGURES_PATH, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado: {filepath}")
    return filepath


@task(name="Graficar Comparación de Modelos")
def plot_model_comparison(df_comparison, substance):
    """
    Genera gráfico comparativo de modelos
    
    Args:
        df_comparison: DataFrame con métricas
        substance: Nombre de la sustancia
    
    Returns:
        str: Ruta del archivo guardado
    """
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Métricas de desempeño
    metrics = ['Accuracy', 'F1 Weighted', 'Precision Macro', 'Recall Macro']
    x = np.arange(len(df_comparison))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0].bar(x + i*width, df_comparison[metric], width, label=metric)
    
    axes[0].set_xlabel('Modelo')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Desempeño de Modelos: {substance}')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(df_comparison['Modelo'], rotation=45)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Tiempo de inferencia
    axes[1].bar(df_comparison['Modelo'], df_comparison['Tiempo Inferencia (s)'])
    axes[1].set_xlabel('Modelo')
    axes[1].set_ylabel('Tiempo (segundos)')
    axes[1].set_title(f'Tiempo de Inferencia: {substance}')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"model_comparison_{substance}.png"
    filepath = os.path.join(FIGURES_PATH, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado: {filepath}")
    return filepath


@task(name="Graficar Feature Importance")
def plot_feature_importance(model, feature_names, substance, model_name):
    """
    Genera gráfico de importancia de features
    
    Args:
        model: Modelo entrenado (debe tener feature_importances_)
        feature_names: Lista de nombres de features
        substance: Nombre de la sustancia
        model_name: Nombre del modelo
    
    Returns:
        str: Ruta del archivo guardado
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️  {model_name} no tiene feature_importances_")
        return None
    
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importancia')
    plt.title(f'Importancia de Features: {substance} - {model_name}')
    plt.tight_layout()
    
    filename = f"feature_importance_{substance}_{model_name}.png"
    filepath = os.path.join(FIGURES_PATH, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado: {filepath}")
    return filepath


@task(name="Analizar Perfiles de Riesgo")
def analyze_risk_profiles(y_true, y_pred, substance):
    """
    Analiza perfiles de riesgo según clasificación
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        substance: Nombre de la sustancia
    
    Returns:
        dict: Análisis de perfiles
    """
    profiles = {
        'Bajo Riesgo (CL0)': {
            'true_count': (y_true == 0).sum(),
            'predicted_count': (y_pred == 0).sum(),
            'correctly_identified': ((y_true == 0) & (y_pred == 0)).sum()
        },
        'Riesgo Moderado (CL1-CL3)': {
            'true_count': ((y_true >= 1) & (y_true <= 3)).sum(),
            'predicted_count': ((y_pred >= 1) & (y_pred <= 3)).sum(),
            'correctly_identified': ((y_true >= 1) & (y_true <= 3) & 
                                   (y_pred >= 1) & (y_pred <= 3)).sum()
        },
        'Alto Riesgo (CL4-CL6)': {
            'true_count': (y_true >= 4).sum(),
            'predicted_count': (y_pred >= 4).sum(),
            'correctly_identified': ((y_true >= 4) & (y_pred >= 4)).sum()
        }
    }
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE PERFILES DE RIESGO: {substance}")
    print(f"{'='*60}")
    
    for profile, data in profiles.items():
        if data['true_count'] > 0:
            recall = data['correctly_identified'] / data['true_count']
            print(f"\n{profile}:")
            print(f"  Casos reales: {data['true_count']}")
            print(f"  Identificados correctamente: {data['correctly_identified']}")
            print(f"  Recall: {recall:.2%}")
    
    return profiles


@task(name="Graficar Perfiles de Riesgo")
def plot_risk_profiles(profiles, substance):
    """
    Genera gráfico de perfiles de riesgo
    
    Args:
        profiles: Dict con análisis de perfiles
        substance: Nombre de la sustancia
    
    Returns:
        str: Ruta del archivo guardado
    """
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    profile_names = list(profiles.keys())
    recalls = []
    
    for profile, data in profiles.items():
        if data['true_count'] > 0:
            recall = data['correctly_identified'] / data['true_count']
            recalls.append(recall * 100)
        else:
            recalls.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(profile_names, recalls, color=['green', 'orange', 'red'])
    plt.xlabel('Perfil de Riesgo')
    plt.ylabel('Recall (%)')
    plt.title(f'Capacidad de Identificación por Perfil de Riesgo: {substance}')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    filename = f"risk_profiles_{substance}.png"
    filepath = os.path.join(FIGURES_PATH, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado: {filepath}")
    return filepath