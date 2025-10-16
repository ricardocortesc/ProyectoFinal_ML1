"""
Utilidades y constantes globales del proyecto
"""

# Nombres de las columnas del dataset
COLUMN_NAMES = [
    "ID", "Age", "Gender", "Education", "Country", "Ethnicity",
    "Nscore", "Escore", "Oscore", "Ascore", "Cscore",
    "Impulsive", "SS",
    "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Chocolate",
    "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh",
    "LSD", "Meth", "Mushrooms", "Nicotine", "Semer", "VSA"
]

# Variables de entrada (features)
FEATURE_COLUMNS = [
    "Age", "Gender", "Education", "Country", "Ethnicity",
    "Nscore", "Escore", "Oscore", "Ascore", "Cscore",
    "Impulsive", "SS"
]

# Variables objetivo (sustancias)
TARGET_COLUMNS = [
    "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Chocolate",
    "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh",
    "LSD", "Meth", "Mushrooms", "Nicotine", "Semer", "VSA"
]

# Mapeo de clases de consumo a valores numéricos
CLASS_MAPPING = {
    "CL0": 0,  # Nunca ha usado
    "CL1": 1,  # Usado hace más de 10 años
    "CL2": 2,  # Usado en la última década
    "CL3": 3,  # Usado en el último año
    "CL4": 4,  # Usado en el último mes
    "CL5": 5,  # Usado en la última semana
    "CL6": 6   # Usado en el último día
}

# Interpretación de las clases
CLASS_LABELS = {
    0: "Nunca",
    1: "Hace +10 años",
    2: "Última década",
    3: "Último año",
    4: "Último mes",
    5: "Última semana",
    6: "Último día"
}

# Rutas del proyecto
DATA_PATH = "data/drug_consumption.csv"
RESULTS_PATH = "results/"
MODELS_PATH = "results/models/"
METRICS_PATH = "results/metrics/"
FIGURES_PATH = "results/figures/"

# Parámetros generales
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # Del train set


def get_consumption_level(code):
    """
    Convierte código de consumo a valor numérico
    
    Args:
        code: Código CL0-CL6 o ya numérico
    
    Returns:
        int: Valor numérico de 0 a 6
    """
    if isinstance(code, str):
        return CLASS_MAPPING.get(code, 0)
    return code


def get_consumption_label(value):
    """
    Convierte valor numérico a etiqueta descriptiva
    
    Args:
        value: Valor numérico de 0 a 6
    
    Returns:
        str: Etiqueta descriptiva
    """
    return CLASS_LABELS.get(value, "Desconocido")