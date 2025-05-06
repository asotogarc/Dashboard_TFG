# Script de Machine Learning para Predicción de Precios de Inmuebles

Este script utiliza la biblioteca **LightGBM** para predecir los precios de inmuebles en varias ciudades españolas, basándose en datos almacenados en archivos `.parquet`. El proceso incluye carga de datos, preprocesamiento, ingeniería de características, optimización de hiperparámetros, entrenamiento del modelo, evaluación y visualización de resultados.

El script está diseñado para ser ejecutado en el mismo directorio donde se encuentra una carpeta `datasets` con los archivos Parquet de datos. Genera gráficos como salida y guarda el modelo entrenado en un archivo.

---


### Sección 3: Parámetros Iniciales

Se define un diccionario con las coordenadas de los centros de las ciudades para calcular distancias más adelante



---

### Sección 4: Carga de Datos desde Parquet

Se define una función para cargar los archivos .parquet desde la carpeta datasets y se cargan los datos de cada ciudad.


---

### Sección 5: Concatenación y Preprocesamiento
Los datos de todas las ciudades se concatenan en un solo DataFrame, asignando un identificador de ciudad a cada registro. Se realiza una limpieza inicial manejando valores nulos y eliminando precios extremos.


---

### Sección 6: Ingeniería de Características
Se crean nuevas características derivadas (como ratios y transformaciones logarítmicas), se procesan las amenidades y se calculan distancias al centro de cada ciudad usando la fórmula de Haversine.


---

### Sección 7: Preparación de Datos para el Modelo
Se seleccionan las características numéricas y categóricas, se dividen los datos en conjuntos de entrenamiento, validación y prueba, y se aplican transformaciones adicionales como normalización.



---

### Sección 8: Optimización de Hiperparámetros con Optuna
Se utiliza Optuna para encontrar los mejores hiperparámetros del modelo LightGBM, optimizando una combinación de métricas y penalizando el sobreajuste.
python


---

### Sección 9: Entrenamiento del Modelo Final
Se entrena el modelo final con los mejores parámetros, combinando los datos de entrenamiento y validación, y se guarda en un archivo.


---

### Sección 10: Evaluación del Modelo
Se evalúa el modelo en el conjunto de prueba, calculando métricas tanto en escala logarítmica como real.


---

### Sección 11: Visualización de Resultados
Se generan gráficos para analizar los resultados, incluyendo la distribución de errores, valores reales vs predichos y la importancia de las características.


---

### Sección 12: Notas Finales
Requisitos: Asegúrate de que la carpeta datasets esté en el directorio raíz del proyecto con los archivos .parquet correspondientes.

Salidas: El script genera archivos de gráficos (error_distribution.png, predicted_vs_actual.png, feature_importance.png, etc.) y el modelo entrenado (lgbm_best_model_improved.txt) en el directorio actual.

Uso: Ejecuta el script en un entorno con las librerías instaladas (puedes usar pip install -r requirements.txt



