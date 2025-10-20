#Procesamiento de los datos

import pandas as pd
from sklearn.model_selection import train_test_split
from prefect import task
from utils import (
    COLUMN_NAMES, 
    FEATURE_COLUMNS, 
    TARGET_COLUMNS,
    CLASS_MAPPING,
    DATA_PATH,
    RANDOM_STATE,
    TEST_SIZE
)


@task(name="Cargar Dataset")
def load_data(filepath=DATA_PATH):
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


@task(name="Limpiar Datos")
def clean_data(df):
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


@task(name="Separar Features y Target")
def get_X_y(df, target_substance):
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


@task(name="Dividir Train/Test")
def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=True):
    stratify_param = y if stratify else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        print(f"Train set: {X_train.shape[0]} muestras")
        print(f"Test set: {X_test.shape[0]} muestras")
        
        return X_train, X_test, y_train, y_test
    
    except ValueError as e:
        print(f"No se puede estratificar (clases insuficientes). Se dividirá sin estratificación")
        return split_data(X, y, test_size, random_state, stratify=False)


@task(name="Preparar Datos para Sustancia")
def prepare_data_for_substance(filepath=DATA_PATH, target_substance="Cannabis"):
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


@task(name="Preparar Todas las Sustancias")
def prepare_all_substances(filepath=DATA_PATH):
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


@task(name="Detectar Top Sustancias Consumidas")
def get_top_consumed_substances(filepath=DATA_PATH, n=5):
    df = load_data(filepath)
    df_clean = clean_data(df)
    
    consumption_scores = {}
    
    for substance in TARGET_COLUMNS:
        # Calcular score de consumo (clases altas valen más)
        y = df_clean[substance]
        # Score = promedio ponderado de las clases
        score = (y > 0).sum()  # Cantidad que consumió al menos una vez
        consumption_scores[substance] = score
    
    # Ordenar por score descendente
    sorted_substances = sorted(consumption_scores.items(), key=lambda x: x[1], reverse=True)
    top_substances = [s[0] for s in sorted_substances[:n]]
    
    print(f"\n{'='*60}")
    print(f"Top {n} sustancias más consumidas:")
    print(f"{'='*60}")
    for i, (substance, score) in enumerate(sorted_substances[:n], 1):
        print(f"{i}. {substance:15} - {score} personas ({score/len(df_clean)*100:.1f}%)")
    
    return top_substances