import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
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

# Revisar la distribución de valores en la columna 'ciudad'
ciudad_counts = df_all['ciudad'].value_counts()
print("Distribución de registros por ciudad:")
print(ciudad_counts)

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

# Añadir características para top amenities
for amenity in top_amenities[:15]:  # Limitamos a los 15 más comunes
    safe_name = amenity.replace(' ', '_').replace('/', '_').lower()
    df_all[f'has_{safe_name}'] = df_all['amenities_list'].apply(lambda x: 1 if amenity in x else 0)

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

# Añadir características de interacción
df_all['accom_dist_interaction'] = df_all['accommodates'] * df_all['log_distance']
df_all['bed_bath_dist'] = df_all['bedrooms'] * df_all['bathrooms'] * df_all['log_distance']
df_all['lux_dist'] = df_all['luxury_score'] * df_all['log_distance']

# Definir características
numeric_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 
    'minimum_nights', 'maximum_nights', 'bathroom_per_person', 
    'bed_to_bedroom_ratio', 'person_per_bedroom', 'person_per_bed','distance_to_center', 
    'log_distance', 'total_amenities', 'luxury_score', 'essential_score',
    'bed_bath_product', 'bed_accom_ratio', 'log_accommodates', 
    'log_minimum_nights', 'log_maximum_nights',
    'accom_dist_interaction', 'bed_bath_dist', 'lux_dist'
] 

# Añadir has_amenity features a las numéricas
amenity_features = [col for col in df_all.columns if col.startswith('has_')]
numeric_features.extend(amenity_features)

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

# Verificar ciudad counts después de filtrado
ciudad_counts_after = df_all['ciudad'].value_counts()
print("Distribución de registros por ciudad después de filtrado:")
print(ciudad_counts_after)

# Eliminar ciudades con muy pocos registros (menos de 5)
min_records_per_city = 5
rare_cities = ciudad_counts_after[ciudad_counts_after < min_records_per_city].index.tolist()
if rare_cities:
    print(f"Eliminando ciudades con menos de {min_records_per_city} registros: {rare_cities}")
    df_all = df_all[~df_all['ciudad'].isin(rare_cities)]
    print(f"Tamaño después de eliminar ciudades raras: {df_all.shape}")

# Preparar datos
X = df_all[numeric_features + categorical_features].copy()
y = np.log1p(df_all['price'])

# Dividir datos - SIN USAR STRATIFY si hay muy pocos datos por ciudad
try:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=df_all['ciudad']
    )
    print("División estratificada realizada con éxito")
except ValueError as e:
    print(f"Error en stratify: {e}")
    print("Realizando división sin stratify")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

# División para validación - también manejamos posibles errores
try:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=X_train_full['ciudad']
    )
except ValueError:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42
    )

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
print(y_test.describe())


# Preparar preprocesadores para XGBoost
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

# Definir función para mean absolute percentage error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

# Entrenar un modelo XGBoost básico primero
print("Entrenando modelo XGBoost base...")

# Preprocesar los datos de entrenamiento
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

base_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42
)

# CORRECCIÓN: Entrenar modelo base - adaptación para versiones antiguas de XGBoost
try:
    # Método para versiones más recientes de XGBoost
    base_xgb.fit(
        X_train_processed,
        y_train,
        eval_set=[(X_val_processed, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
except TypeError:
    # Método alternativo para versiones muy antiguas
    dtrain = xgb.DMatrix(X_train_processed, label=y_train)
    dval = xgb.DMatrix(X_val_processed, label=y_val)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    evallist = [(dtrain, 'train'), (dval, 'validation')]
    
    bst = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100,
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Asignar el modelo entrenado al regresor
    base_xgb._Booster = bst

# Evaluar en conjunto de validación
y_val_pred = base_xgb.predict(X_val_processed)
val_mse = mean_squared_error(y_val, y_val_pred)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Evaluación del modelo base (validación):")
print(f"  MSE: {val_mse:.6f}")
print(f"  MAPE: {val_mape:.2f}%")
print(f"  R2: {val_r2:.6f}")

# Buscar mejores hiperparámetros con GridSearchCV
print("\nAfinando hiperparámetros con GridSearchCV...")

# Definir pipeline completo para GridSearchCV
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgboost', xgb.XGBRegressor(random_state=42, tree_method='hist'))
])

# Parámetros para búsqueda en grid
param_grid = {
    'xgboost__n_estimators': [150, 200],
    'xgboost__max_depth': [5, 7, 9],
    'xgboost__learning_rate': [0.05, 0.1],
    'xgboost__subsample': [0.8, 0.9],
    'xgboost__colsample_bytree': [0.8, 0.9],
    'xgboost__reg_alpha': [0.1, 1],
    'xgboost__reg_lambda': [0.1, 1]
}

# GridSearchCV con 3 folds
grid_search = GridSearchCV(
    xgb_pipeline,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Para ahorrar tiempo, limitamos a una muestra más pequeña
sample_size = min(100000, len(X_train))
X_train_sample = X_train.sample(sample_size, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

print(f"Realizando GridSearchCV con {sample_size} muestras...")
grid_search.fit(X_train_sample, y_train_sample)

print(f"\nMejores parámetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Entrenar modelo final con los mejores parámetros
print("\nEntrenando modelo final con los mejores parámetros en todos los datos de entrenamiento...")

# Crear pipeline con los mejores parámetros
best_params = {k.replace('xgboost__', ''): v for k, v in grid_search.best_params_.items()}
final_xgb = xgb.XGBRegressor(
    **best_params,
    random_state=42,
    tree_method='hist'
)

# Preprocesar todos los datos de entrenamiento
X_train_full_processed = preprocessor.fit_transform(X_train_full)
X_test_processed = preprocessor.transform(X_test)

# CORRECCIÓN: Entrenar modelo final - compatible con versiones antiguas
try:
    # Intento normal
    final_xgb.fit(X_train_full_processed, y_train_full)
except Exception as e:
    print(f"Error en el entrenamiento estándar: {e}")
    print("Intentando con la API de bajo nivel...")
    
    # Método alternativo con la API de bajo nivel
    dtrain_full = xgb.DMatrix(X_train_full_processed, label=y_train_full)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': best_params.get('max_depth', 6),
        'eta': best_params.get('learning_rate', 0.1),
        'subsample': best_params.get('subsample', 0.8),
        'colsample_bytree': best_params.get('colsample_bytree', 0.8),
        'reg_alpha': best_params.get('reg_alpha', 0),
        'reg_lambda': best_params.get('reg_lambda', 1)
    }
    
    num_boost = best_params.get('n_estimators', 200)
    bst = xgb.train(params, dtrain_full, num_boost_round=num_boost)
    
    # Asignar el modelo entrenado al regresor
    final_xgb._Booster = bst

# Guardar el modelo y el preprocesador
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_xgb)
])
model_path = 'xgboost_model_pipeline.pkl'
joblib.dump(model_pipeline, model_path)
print(f"Modelo y preprocesador guardados en {model_path}")

# Evaluar en conjunto de prueba
print("\nEvaluando modelo en conjunto de prueba...")
y_test_pred = final_xgb.predict(X_test_processed)

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
feature_importance = final_xgb.feature_importances_

# Obtener nombres de características después del preprocesamiento
cat_cols_names = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_cols_names)

# Verificar si las dimensiones coinciden
if len(feature_importance) == len(all_feature_names):
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 30 características más importantes:")
    print(importance_df.head(30))
else:
    print("\nNo se pueden mostrar las importancias debido a diferencias en las dimensiones")
    print(f"Tamaño de importancias: {len(feature_importance)}")
    print(f"Tamaño de nombres de características: {len(all_feature_names)}")

# Visualizar algunas predicciones
print("\nAlgunas predicciones:")
sample_size = min(10, len(y_test))
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
sample_y_test = np.array(y_test)[sample_indices]
sample_y_pred = y_test_pred[sample_indices]

# Convertir a escala real para mejor interpretación
sample_y_test_real = np.expm1(sample_y_test)
sample_y_pred_real = np.expm1(sample_y_pred)

for i, (true, pred) in enumerate(zip(sample_y_test_real, sample_y_pred_real)):
    print(f"Muestra {i+1}: Real: {true:.2f}, Predicción: {pred:.2f}, Error: {abs(true-pred):.2f}")

# Graficar importancia de características (top 15)
if len(feature_importance) == len(all_feature_names):
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 características más importantes')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    print("\nGráfico de importancia de características guardado como 'xgboost_feature_importance.png'")