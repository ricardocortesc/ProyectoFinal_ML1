"""
Módulo de carga y preprocesamiento de datos
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from utils import (
    COLUMN_NAMES, 
    FEATURE_COLUMNS, 
    TARGET_COLUMNS,
    CLASS_MAPPING,
    DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE
)


def load_data(filepath=DATA_PATH):
    """
    Carga el dataset desde CSV
    
    Args:
        filepath: Ruta al archivo CSV
    
    Returns:
        pd.DataFrame: Dataset cargado con nombres de columnas
    """
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df):
    """
    Limpia y preprocesa el dataset
    
    Args:
        df: DataFrame original
    
    Returns:
        pd.DataFrame: Dataset limpio
    """
    df_clean = df.copy()
    
    # Eliminar columna ID
    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['ID'])
    
    # Convertir códigos CL a valores numéricos
    for col in TARGET_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(CLASS_MAPPING)
    
    print(f"Datos limpiados: {df_clean.shape}")
    return df_clean


def get_X_y(df, target_substance):
    """
    Separa features (X) y target (y) para una sustancia específica
    
    Args:
        df: DataFrame limpio
        target_substance: Nombre de la sustancia a predecir
    
    Returns:
        tuple: (X, y) - Features y target
    """
    if target_substance not in TARGET_COLUMNS:
        raise ValueError(f"Sustancia '{target_substance}' no encontrada en el dataset")
    
    # Features: todas las columnas excepto las sustancias
    X = df[FEATURE_COLUMNS].copy()
    
    # Target: la sustancia específica
    y = df[target_substance].copy()
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Distribución de clases para {target_substance}:")
    print(y.value_counts().sort_index())
    
    return X, y


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=True):
    """
    Divide datos en train y test
    
    Args:
        X: Features
        y: Target
        test_size: Proporción de test
        random_state: Semilla aleatoria
        stratify: Si usar estratificación
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"Train set: {X_train.shape[0]} muestras")
    print(f"Test set: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test


def prepare_data_for_substance(filepath=DATA_PATH, target_substance="Cannabis"):
    """
    Pipeline completo de preparación de datos para una sustancia
    
    Args:
        filepath: Ruta al CSV
        target_substance: Sustancia a predecir
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\n{'='*60}")
    print(f"Preparando datos para: {target_substance}")
    print(f"{'='*60}\n")
    
    # Cargar y limpiar
    df = load_data(filepath)
    df_clean = clean_data(df)
    
    # Separar X e y
    X, y = get_X_y(df_clean, target_substance)
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    return X_train, X_test, y_train, y_test


def prepare_all_substances(filepath=DATA_PATH):
    """
    Prepara datos para todas las sustancias
    
    Args:
        filepath: Ruta al CSV
    
    Returns:
        dict: Diccionario con datos divididos por sustancia
    """
    df = load_data(filepath)
    df_clean = clean_data(df)
    
    all_data = {}
    
    for substance in TARGET_COLUMNS:
        print(f"\nProcesando: {substance}")
        X, y = get_X_y(df_clean, substance)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        all_data[substance] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    return all_data