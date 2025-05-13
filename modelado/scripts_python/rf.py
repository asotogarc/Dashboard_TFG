import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import Counter
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
warnings.filterwarnings('ignore')

# Definir centros de ciudades
ciudad_centros = {
    1: (40.4168, -3.7038), 2: (41.3879, 2.1700), 3: (37.3772, -5.9869),
    4: (39.4699, -0.3763), 5: (36.7202, -4.4214), 6: (39.8885, 4.2658),
    7: (39.5696, 2.6501), 8: (41.9842, 2.8214), 9: (43.2627, -2.9253)
}

# Función para cargar archivos parquet
def load_parquet(file_name):
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    file_path = project_root / 'datasets' / file_name
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error al cargar {file_name}: {e}")
        return pd.DataFrame()

# Cargar datos
df_madrid = load_parquet('inmuebles_madrid.parquet')
df_barcelona = load_parquet('inmuebles_barcelona.parquet')
df_euskadi = load_parquet('inmuebles_euskadi.parquet')
df_sevilla = load_parquet('inmuebles_sevilla.parquet')
df_valencia = load_parquet('inmuebles_valencia.parquet')
df_malaga = load_parquet('inmuebles_málaga.parquet')
df_menorca = load_parquet('inmuebles_menorca.parquet')
df_mallorca = load_parquet('inmuebles_mallorca.parquet')
df_girona = load_parquet('inmuebles_girona.parquet')

# Combinar datasets
df_all = pd.concat([
    df_madrid.assign(ciudad=1), df_barcelona.assign(ciudad=2),
    df_sevilla.assign(ciudad=3), df_valencia.assign(ciudad=4),
    df_malaga.assign(ciudad=5), df_menorca.assign(ciudad=6),
    df_mallorca.assign(ciudad=7), df_girona.assign(ciudad=8),
    df_euskadi.assign(ciudad=9)
]).reset_index(drop=True)

print(f"Tamaño total del conjunto de datos: {df_all.shape}")

# Imputar valores faltantes en columnas numéricas
for col in ['bathrooms', 'bedrooms', 'beds', 'accommodates']:
    if df_all[col].isnull().any():
        df_all[col].fillna(df_all[col].median(), inplace=True)

# Filtrar precios mayores a 0
df_all = df_all[df_all['price'] > 0]

# Eliminar outliers de precio por grupo
def remove_price_outliers(df, lower_percentile=0.01, upper_percentile=0.99):
    groups = df.groupby(['ciudad', 'property_type'])
    filtered_dfs = []
    for name, group in groups:
        if len(group) >= 20:
            Q1, Q3 = group['price'].quantile([lower_percentile, upper_percentile])
            filtered_group = group[(group['price'] >= Q1) & (group['price'] <= Q3)]
            filtered_dfs.append(filtered_group)
        else:
            filtered_dfs.append(group)
    return pd.concat(filtered_dfs).reset_index(drop=True)

df_all = remove_price_outliers(df_all, 0.005, 0.995)
print(f"Tamaño después de filtrar outliers: {df_all.shape}")

# Crear nuevas características
df_all['bathroom_per_person'] = df_all['bathrooms'] / df_all['accommodates'].replace(0, 1)
df_all['bed_to_bedroom_ratio'] = df_all['beds'] / df_all['bedrooms'].replace(0, 1)
df_all['person_per_bedroom'] = df_all['accommodates'] / df_all['bedrooms'].replace(0, 1)
df_all['person_per_bed'] = df_all['accommodates'] / df_all['beds'].replace(0, 1)
df_all['bed_bath_product'] = df_all['beds'] * df_all['bathrooms']
df_all['bed_accom_ratio'] = df_all['beds'] / df_all['accommodates'].replace(0, 1)

# Transformaciones logarítmicas
for col in ['minimum_nights', 'maximum_nights', 'accommodates']:
    df_all[f'log_{col}'] = np.log1p(df_all[col])

# Procesar amenities
df_all['amenities_list'] = df_all['amenities'].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) and x != '[]' else []
)
df_all['total_amenities'] = df_all['amenities_list'].apply(len)

# Identificar top amenities
all_amenities = [amenity for sublist in df_all['amenities_list'] for amenity in sublist]
top_amenities = [amenity for amenity, count in Counter(all_amenities).most_common(30) 
                 if count > len(df_all) * 0.05]

# Calcular scores de amenities
luxury_amenities = ['Pool', 'Hot tub', 'Gym', 'Doorman', 'Elevator']
essential_amenities = ['Wifi', 'Kitchen', 'Heating', 'Air conditioning', 'Washer']
df_all['luxury_score'] = df_all['amenities_list'].apply(
    lambda x: sum(1 for item in luxury_amenities if item in x) / len(luxury_amenities)
)
df_all['essential_score'] = df_all['amenities_list'].apply(
    lambda x: sum(1 for item in essential_amenities if item in x) / len(essential_amenities)
)

# Clustering de vecindarios
def apply_kmeans(group, n_clusters=15):
    if len(group) > n_clusters:
        coords = group[['latitude', 'longitude']].values
        kmeans = MiniBatchKMeans(n_clusters=min(n_clusters, len(group)), 
                                random_state=42, batch_size=1000,
                                init='k-means++', max_iter=300)
        return kmeans.fit_predict(coords)
    return np.zeros(len(group), dtype=int)

df_all['neighborhood_cluster'] = df_all.groupby('ciudad', group_keys=False).apply(
    lambda group: pd.Series(apply_kmeans(group, n_clusters=20), index=group.index)
)

# Calcular distancia al centro
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

city_centers = df_all['ciudad'].map(ciudad_centros)
df_all['center_lat'] = city_centers.apply(lambda x: x[0])
df_all['center_lon'] = city_centers.apply(lambda x: x[1])
df_all['distance_to_center'] = haversine_vectorized(
    df_all['latitude'], df_all['longitude'], df_all['center_lat'], df_all['center_lon']
)
df_all['log_distance'] = np.log1p(df_all['distance_to_center'])

# Definir características
numeric_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 
    'minimum_nights', 'maximum_nights', 'bathroom_per_person', 
    'bed_to_bedroom_ratio', 'person_per_bedroom', 'person_per_bed','distance_to_center', 
    'log_distance', 'total_amenities', 'luxury_score', 'essential_score',
    'bed_bath_product', 'bed_accom_ratio', 'log_accommodates', 
    'log_minimum_nights', 'log_maximum_nights'
] 

categorical_features = ['property_type', 'room_type', 'ciudad', 'neighborhood_cluster']

all_features = numeric_features + categorical_features

# Verificar columnas faltantes
missing_cols = [col for col in all_features if col not in df_all.columns]
if missing_cols:
    print(f"Columnas faltantes: {missing_cols}")
    numeric_features = [f for f in numeric_features if f in df_all.columns]
    categorical_features = [f for f in categorical_features if f in df_all.columns]

# Contar el número de filas antes de eliminar
filas_antes = df_all.shape[0]

# Eliminar las filas donde price > 1000
df_all = df_all[df_all['price'] <= 100]

# Contar el número de filas después de eliminar
filas_despues = df_all.shape[0]

# Calcular e imprimir el número de filas eliminadas
filas_eliminadas = filas_antes - filas_despues
print(f"Se han eliminado {filas_eliminadas} filas con precio superior a 1000.")

print(df_all['price'].describe())

# Preparar datos
X = df_all[numeric_features + categorical_features].copy()
y = np.log1p(df_all['price'])
print(y.describe())


# Dividir datos
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=df_all['ciudad']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=X_train_full['ciudad']
)
print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# Preparar preprocesadores para RandomForest

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocesador columnar
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Definir una función para mean absolute percentage error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

# Realizar grid search para RandomForest
print("Iniciando búsqueda de hiperparámetros para Random Forest...")

# Definir parámetros iniciales para RandomForest
rf_params = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1  # Usar todos los núcleos disponibles
}

# Para limitar el uso de memoria podemos usar un RandomForest con menos árboles inicialmente
print("Entrenando modelo base...")
base_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

# Pipeline completo
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', base_rf)
])

# Entrenar en el conjunto de entrenamiento
pipe.fit(X_train, y_train)

# Evaluar en conjunto de validación
y_val_pred = pipe.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Evaluación del modelo base (validación):")
print(f"  MSE: {val_mse:.6f}")
print(f"  MAPE: {val_mape:.2f}%")
print(f"  R2: {val_r2:.6f}")

# Intentamos mejorar el modelo con un ajuste manual de hiperparámetros
print("\nEntrenando modelo mejorado con hiperparámetros ajustados...")

improved_rf = RandomForestRegressor(
    n_estimators=200,        # Más árboles para mejor estabilidad
    max_depth=15,            # Profundidad incrementada pero controlada
    min_samples_split=15,    # Reducido para permitir más divisiones
    min_samples_leaf=4,      # Reducido para mayor especificidad
    max_features='sqrt',     # Típicamente funciona bien
    bootstrap=True,          # Usar bootstrapping para reducir sobreajuste
    random_state=42,
    n_jobs=-1
)

improved_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', improved_rf)
])

# Entrenar modelo mejorado
improved_pipe.fit(X_train, y_train)

# Evaluar modelo mejorado en validación
y_val_pred_improved = improved_pipe.predict(X_val)
val_mse_improved = mean_squared_error(y_val, y_val_pred_improved)
val_mape_improved = mean_absolute_percentage_error(y_val, y_val_pred_improved)
val_r2_improved = r2_score(y_val, y_val_pred_improved)

print(f"Evaluación del modelo mejorado (validación):")
print(f"  MSE: {val_mse_improved:.6f}")
print(f"  MAPE: {val_mape_improved:.2f}%")
print(f"  R2: {val_r2_improved:.6f}")

# Entrenar modelo final con todos los datos de entrenamiento
print("\nEntrenando modelo final con todos los datos de entrenamiento...")

X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

final_rf = RandomForestRegressor(
    n_estimators=250,        # Incluso más árboles para el modelo final
    max_depth=15,
    min_samples_split=15,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

final_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_rf)
])

final_pipe.fit(X_train_full, y_train_full)

# Guardar el modelo
model_path = 'random_forest_model.pkl'
joblib.dump(final_pipe, model_path)
print(f"Modelo guardado en {model_path}")

# Evaluar en conjunto de prueba
print("\nEvaluando modelo en conjunto de prueba...")
y_test_pred = final_pipe.predict(X_test)

# Métricas en escala logarítmica
rmse_log = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_log = mean_absolute_error(y_test, y_test_pred)
r2_log = r2_score(y_test, y_test_pred)
print(f"Métricas (escala logarítmica):")
print(f"    RMSE: {rmse_log:.4f}")
print(f"    MAE: {mae_log:.4f}")
print(f"    R²: {r2_log:.4f}")

# Transformar a escala real
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_test_pred)

# Métricas en escala real
rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae_real = mean_absolute_error(y_test_real, y_pred_real)
r2_real = r2_score(y_test_real, y_pred_real)
mape = np.mean(np.abs((y_test_real - y_pred_real) / np.maximum(y_test_real, 1))) * 100
print(f"\nMétricas (escala real):")
print(f"    RMSE: {rmse_real:.2f}")
print(f"    MAE: {mae_real:.2f}")
print(f"    R²: {r2_real:.4f}")
print(f"    MAPE: {mape:.2f}%")

# Analizar importancia de características
# Obtener los nombres de características después del preprocesamiento
cat_features = list(preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_features

# Obtener importancias - siendo cuidadosos con la dimensionalidad después del preprocesamiento
model_rf = final_pipe.named_steps['model']
importances = model_rf.feature_importances_

# Asegurarnos de que coincidan las dimensiones
if len(importances) == len(feature_names):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 30 características más importantes:")
    print(importance_df.head(30))
else:
    print("\nNo se pueden mostrar las importancias debido a diferencias en las dimensiones")
    print(f"Tamaño de importancias: {len(importances)}")
    print(f"Tamaño de nombres de características: {len(feature_names)}")

# Visualizar algunas predicciones
print("\nAlgunas predicciones:")
sample_size = min(10, len(y_test))
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
sample_y_test = y_test.iloc[sample_indices]
sample_y_pred = y_test_pred[sample_indices]

# Convertir a escala real para mejor interpretación
sample_y_test_real = np.expm1(sample_y_test)
sample_y_pred_real = np.expm1(sample_y_pred)

for i, (true, pred) in enumerate(zip(sample_y_test_real, sample_y_pred_real)):
    print(f"Muestra {i+1}: Real: {true:.2f}, Predicción: {pred:.2f}, Error: {abs(true-pred):.2f}")