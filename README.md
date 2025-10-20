# Proyecto Final - Machine Learning 1
- Sebastián Castellanos Estupiñan
- Ricardo Andrés Cortés Coronell
- Andrés Felipe Sanchez Rincón

**Modo de ejecución**
- Ejecutar el pipeline.py directamente

**Modo de Ejecución Recomendable**
- Anaconda Prompt "conda activate <nombre_ambiente>"
- Anaconda Prompt "cd <ruta_archivo>"
- Aanaconda Prompt "python pipeline.py"

**Flujos de "pipeline.py"**

Este incluye tres flujos comentados, solo es quitar el # al que se quiera ejecutar.
- Pipeline: 5 sustancias más consumidas
  #results = drug_prediction_top_substances(n_substances=5)
    
- Pipeline: Una sola sustancia
  #result = drug_prediction_single_substance("Cannabis")

- Pipeline: Todas las sustancias:
  #results = drug_prediction_multiple_substances()
