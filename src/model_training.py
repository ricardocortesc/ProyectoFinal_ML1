import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from prefect import task
from utils import MODELS_PATH, RANDOM_STATE
import os


@task(name="Entrenar Random Forest")
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Entrena un modelo Random Forest
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        n_estimators: Número de árboles
        max_depth: Profundidad máxima
    
    Returns:
        tuple: (modelo, tiempo_entrenamiento)
    """
    print(f"\nEntrenando Random Forest...")
    print(f"Parámetros: n_estimators={n_estimators}, max_depth={max_depth}")
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Entrenamiento completado en {training_time:.2f} segundos")
    
    return model, training_time


@task(name="Entrenar XGBoost")
def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Entrena un modelo XGBoost
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        n_estimators: Número de estimadores
        max_depth: Profundidad máxima
        learning_rate: Tasa de aprendizaje
    
    Returns:
        tuple: (modelo, tiempo_entrenamiento)
    """
    print(f"\nEntrenando XGBoost...")
    print(f"Parámetros: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
    
    start_time = time.time()
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Entrenamiento completado en {training_time:.2f} segundos")
    
    return model, training_time


@task(name="Entrenar Regresión Logística")
def train_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Entrena un modelo de Regresión Logística multinomial
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        max_iter: Iteraciones máximas
    
    Returns:
        tuple: (modelo, tiempo_entrenamiento)
    """
    print(f"\nEntrenando Regresión Logística...")
    print(f"Parámetros: max_iter={max_iter}")
    
    start_time = time.time()
    
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        multi_class='multinomial',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Entrenamiento completado en {training_time:.2f} segundos")
    
    return model, training_time


@task(name="Guardar Modelo")
def save_model(model, substance, model_name):
    """
    Guarda un modelo entrenado
    
    Args:
        model: Modelo entrenado
        substance: Nombre de la sustancia
        model_name: Nombre del algoritmo
    
    Returns:
        str: Ruta donde se guardó el modelo
    """
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    filename = f"{substance}_{model_name}.pkl"
    filepath = os.path.join(MODELS_PATH, filename)
    
    joblib.dump(model, filepath)
    print(f"Modelo guardado en: {filepath}")
    
    return filepath


@task(name="Cargar Modelo")
def load_model(filepath):
    """
    Carga un modelo guardado
    
    Args:
        filepath: Ruta al archivo del modelo
    
    Returns:
        modelo: Modelo cargado
    """
    model = joblib.load(filepath)
    print(f"Modelo cargado desde: {filepath}")
    return model


@task(name="Entrenar Múltiples Modelos")
def train_all_models(X_train, y_train, substance):
    """
    Entrena todos los modelos disponibles para una sustancia
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        substance: Nombre de la sustancia
    
    Returns:
        dict: Diccionario con modelos y tiempos de entrenamiento
    """
    results = {}
    
    # Random Forest
    rf_model, rf_time = train_random_forest(X_train, y_train)
    save_model(rf_model, substance, "RandomForest")
    results['RandomForest'] = {'model': rf_model, 'training_time': rf_time}
    
    # XGBoost
    xgb_model, xgb_time = train_xgboost(X_train, y_train)
    save_model(xgb_model, substance, "XGBoost")
    results['XGBoost'] = {'model': xgb_model, 'training_time': xgb_time}
    
    # Logistic Regression
    lr_model, lr_time = train_logistic_regression(X_train, y_train)
    save_model(lr_model, substance, "LogisticRegression")
    results['LogisticRegression'] = {'model': lr_model, 'training_time': lr_time}
    
    print(f"\n{'='*60}")
    print(f"Resumen de entrenamientos para {substance}:")
    print(f"{'='*60}")
    for name, info in results.items():
        print(f"{name}: {info['training_time']:.2f}s")
    
    return results