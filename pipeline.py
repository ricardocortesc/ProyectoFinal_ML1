#Flujo Principal del Proyecto

from prefect import flow
from src.data_processing import (
    load_data,
    clean_data,
    get_X_y,
    split_data,
    get_top_consumed_substances
)
from src.model_training import train_all_models
from src.evaluation import (
    evaluate_all_models,
    compare_models,
    save_metrics,
    get_best_model
)
from src.visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance,
    analyze_risk_profiles,
    plot_risk_profiles
)
from utils import DATA_PATH, TARGET_COLUMNS, FEATURE_COLUMNS


@flow(name="Pipeline de Predicción - Una Sustancia")
def drug_prediction_single_substance(substance="Cannabis"):
    print(f"\n{'#'*80}")
    print(f"# INICIANDO PIPELINE PARA: {substance}")
    print(f"{'#'*80}\n")
    
    # 1. Cargar y preparar datos
    df = load_data(DATA_PATH)
    df_clean = clean_data(df)
    X, y = get_X_y(df_clean, substance)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 2. Entrenar modelos
    models_dict = train_all_models(X_train, y_train, substance)
    
    # 3. Evaluar modelos
    evaluation_results = evaluate_all_models(models_dict, X_test, y_test, substance)
    
    # 4. Comparar y guardar resultados
    df_comparison = compare_models(evaluation_results)
    save_metrics(df_comparison, f"{substance}_comparison.csv")
    
    # 5. Identificar mejor modelo
    best_model = get_best_model(df_comparison)
    best_result = [r for r in evaluation_results if r['model_name'] == best_model['Modelo']][0]
    
    # 6. Generar visualizaciones
    plot_model_comparison(df_comparison, substance)
    plot_confusion_matrix(best_result['confusion_matrix'], substance, best_model['Modelo'])
    plot_feature_importance(best_result['model'], FEATURE_COLUMNS, substance, best_model['Modelo'])
    
    # 7. Analizar perfiles de riesgo
    profiles = analyze_risk_profiles(y_test, best_result['y_pred'], substance)
    plot_risk_profiles(profiles, substance)
    
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETADO PARA: {substance}")
    print(f"{'#'*80}\n")
    
    return {
        'substance': substance,
        'models': models_dict,
        'evaluation': evaluation_results,
        'comparison': df_comparison,
        'best_model': best_model,
        'risk_profiles': profiles
    }


@flow(name="Pipeline de Predicción - Múltiples Sustancias")
def drug_prediction_multiple_substances(substances=None):
    if substances is None:
        substances = TARGET_COLUMNS
    
    print(f"\n{'#'*80}")
    print(f"# INICIANDO PIPELINE PARA {len(substances)} SUSTANCIAS")
    print(f"{'#'*80}\n")
    
    all_results = {}
    
    # Cargar y limpiar datos una sola vez
    df = load_data(DATA_PATH)
    df_clean = clean_data(df)
    
    # Procesar cada sustancia
    for substance in substances:
        print(f"\n{'='*80}")
        print(f"Procesando: {substance}")
        print(f"{'='*80}\n")
        
        # Preparar datos
        X, y = get_X_y(df_clean, substance)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Entrenar modelos
        models_dict = train_all_models(X_train, y_train, substance)
        
        # Evaluar modelos
        evaluation_results = evaluate_all_models(models_dict, X_test, y_test, substance)
        
        # Comparar resultados
        df_comparison = compare_models(evaluation_results)
        save_metrics(df_comparison, f"{substance}_comparison.csv")
        
        # Visualizaciones
        plot_model_comparison(df_comparison, substance)
        
        # Mejor modelo
        best_model = get_best_model(df_comparison)
        best_result = [r for r in evaluation_results if r['model_name'] == best_model['Modelo']][0]
        
        plot_confusion_matrix(best_result['confusion_matrix'], substance, best_model['Modelo'])
        plot_feature_importance(best_result['model'], FEATURE_COLUMNS, substance, best_model['Modelo'])
        
        # Perfiles de riesgo
        profiles = analyze_risk_profiles(y_test, best_result['y_pred'], substance)
        plot_risk_profiles(profiles, substance)
        
        # Guardar resultados
        all_results[substance] = {
            'comparison': df_comparison,
            'evaluation': evaluation_results,
            'risk_profiles': profiles
        }
    
    # Crear resumen global
    print(f"\n{'#'*80}")
    print(f"# RESUMEN GLOBAL - {len(substances)} SUSTANCIAS PROCESADAS")
    print(f"{'#'*80}\n")
    
    for substance, results in all_results.items():
        best = results['comparison'].loc[results['comparison']['F1 Weighted'].idxmax()]
        print(f"{substance:15} | Mejor: {best['Modelo']:20} | F1: {best['F1 Weighted']:.4f} | Acc: {best['Accuracy']:.4f}")
    
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETADO PARA TODAS LAS SUSTANCIAS")
    print(f"{'#'*80}\n")
    
    return all_results


@flow(name="Pipeline de Predicción - Top Sustancias")
def drug_prediction_top_substances(n_substances=5):
    # Detectar automáticamente las más consumidas
    top_substances = get_top_consumed_substances(DATA_PATH, n_substances)
    
    return drug_prediction_multiple_substances(top_substances)


if __name__ == "__main__":
    # Pipeline: 5 sustancias más consumidas
    #results = drug_prediction_top_substances(n_substances=5)
    
    # Pipeline: Una sola sustancia
    result = drug_prediction_single_substance("Cannabis")

    # Pipeline: Todas las sustancias:
    #results = drug_prediction_multiple_substances()