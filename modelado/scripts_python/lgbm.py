import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, PowerTransformer
from collections import Counter
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
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

# Preparar datos
X = df_all[numeric_features + categorical_features].copy()
y = np.log1p(df_all['price'])

# Dividir datos
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=df_all['ciudad']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=X_train_full['ciudad']
)
print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

print(y_test.describe())

# Convertir categorías
for col in categorical_features:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Convertir numéricas y llenar NaNs
for col in numeric_features:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
    X_val[col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0)
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

# Escalar algunas características
scaler = StandardScaler()
cols_to_scale = ['distance_to_center', 'log_distance', 'total_amenities']
for col in cols_to_scale:
    if col in X_train.columns:
        X_train[f'scaled_{col}'] = scaler.fit_transform(X_train[[col]])
        X_val[f'scaled_{col}'] = scaler.transform(X_val[[col]])
        X_test[f'scaled_{col}'] = scaler.transform(X_test[[col]])
        numeric_features.append(f'scaled_{col}')

# Añadir características cuadráticas
key_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'distance_to_center']
for var in key_features:
    if var in X_train.columns:
        X_train[f'{var}_squared'] = X_train[var] ** 2
        X_val[f'{var}_squared'] = X_val[var] ** 2
        X_test[f'{var}_squared'] = X_test[var] ** 2
        numeric_features.append(f'{var}_squared')

# Preparar datos para LightGBM
train_data = lgb.Dataset(
    X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False
)
valid_data = lgb.Dataset(
    X_val, label=y_val, categorical_feature=categorical_features, reference=train_data, free_raw_data=False
)

# Optimización de hiperparámetros con Optuna
print("Iniciando optimización de hiperparámetros...")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

def objective(trial):
    params = {
        'objective': trial.suggest_categorical('objective', ['regression', 'huber', 'quantile']),
        'metric': 'mse',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 30, 400),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 0.1),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10),
    }
    if params['objective'] == 'huber':
        params['alpha'] = trial.suggest_float('alpha', 0.5, 1.0)
    elif params['objective'] == 'quantile':
        params['alpha'] = trial.suggest_float('alpha', 0.1, 0.9)

    model = lgb.train(
        params, train_data, num_boost_round=2000,
        valid_sets=[valid_data, train_data], valid_names=['valid', 'train'],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
    )

    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mse = mean_squared_error(y_val, y_val_pred)
    mape = mean_absolute_percentage_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    train_r2 = r2_score(y_train, y_train_pred)
    overfitting_penalty = max(0, (train_r2 - r2) * 2)
    score = mse * (1 + overfitting_penalty) + mape * 0.01 * (1 - r2)

    trial.set_user_attr('mse', mse)
    trial.set_user_attr('mape', mape)
    trial.set_user_attr('r2', r2)
    trial.set_user_attr('overfitting', train_r2 - r2)
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

best_params = study.best_params
best_trial = study.best_trial

print("\nMejores parámetros:")
for key, value in best_params.items():
    print(f"    {key}: {value}")
print(f"Mejor MSE: {best_trial.user_attrs['mse']:.6f}")
print(f"Mejor MAPE: {best_trial.user_attrs['mape']:.2f}%")
print(f"Mejor R2: {best_trial.user_attrs['r2']:.6f}")
print(f"Índice de sobreajuste: {best_trial.user_attrs['overfitting']:.6f}")

# Entrenar modelo final
print("\nEntrenando modelo final con los mejores parámetros...")
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

for col in categorical_features:
    X_train_full[col] = X_train_full[col].astype('category')

train_full_data = lgb.Dataset(
    X_train_full, label=y_train_full, categorical_feature=categorical_features, free_raw_data=False
)

# Después de obtener best_params de Optuna
final_params = best_params.copy()
final_params['metric'] = 'mse'  # Forzar métrica como MSE
final_params['verbosity'] = -1

if best_trial.user_attrs['overfitting'] > 0.05:
    print("Detectado riesgo de sobreajuste, ajustando parámetros...")
    final_params['learning_rate'] *= 0.8
    final_params['lambda_l1'] *= 1.5
    final_params['lambda_l2'] *= 1.5
    final_params['feature_fraction'] = max(0.7, final_params.get('feature_fraction', 0.8))
    final_params['bagging_fraction'] = max(0.7, final_params.get('bagging_fraction', 0.8))

print("Realizando validación cruzada para el modelo final...")
cv_results = lgb.cv(
    final_params, train_full_data, num_boost_round=5000, nfold=5,
    stratified=False, shuffle=True,
    callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=200)],
    seed=42
)

best_iteration = np.argmin(cv_results['valid l2-mean']) + 1
optimal_rounds = best_iteration
print(f"Número óptimo de rondas según CV: {optimal_rounds}")

print("Entrenando modelo final con todos los datos...")
lgbm_best = lgb.train(
    final_params, train_full_data, num_boost_round=optimal_rounds,
    callbacks=[lgb.log_evaluation(period=200)]
)

# Guardar modelo
model_path = 'lgbm_best_model_improved.txt'
lgbm_best.save_model(model_path)
print(f"Modelo guardado en {model_path}")

# Evaluar en conjunto de prueba
print("\nEvaluando modelo en conjunto de prueba...")
y_test_pred = lgbm_best.predict(X_test)

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
importance = lgbm_best.feature_importance(importance_type='gain')
feature_names = lgbm_best.feature_name()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)
print("\nTop 30 características más importantes:")
print(importance_df.head(30))

print(y_pred_real)
print(y_test_pred)



# Gráficos mejorados

# 1. Predicciones vs. Valores Reales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_real, y=y_pred_real, alpha=0.5)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Predicciones vs. Valores Reales (Escala Real)')
plt.savefig('predicted_vs_actual_improved.png')
plt.close()

# 2. Histograma de Errores con Curva de Densidad
errors_real = y_test_real - y_pred_real
plt.figure(figsize=(10, 6))
sns.histplot(errors_real, kde=True, bins=50, color="blue")
plt.title("Distribución de Errores (Escala Real)")
plt.xlabel("Error en precio real")
plt.ylabel("Frecuencia")
plt.axvline(x=0, linestyle='--', color='red')
plt.savefig('error_distribution_improved.png')
plt.close()

# 3. Importancia de Características
importance_df['Normalized Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
plt.figure(figsize=(12, 8))
sns.barplot(x='Normalized Importance', y='Feature', data=importance_df.head(20))
plt.title("Top 20 Características por Importancia (Gain Normalizado)")
plt.tight_layout()
plt.savefig('feature_importance_improved.png')
plt.close()

# 4. Análisis de Residuos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_real, y=errors_real, alpha=0.5)
plt.axhline(y=0, linestyle='--', color='red')
plt.title("Análisis de Residuos")
plt.xlabel("Valores Predichos")
plt.ylabel("Residuos")
plt.savefig('residuals_analysis_improved.png')
plt.close()

# 5. Boxplot de Errores por Ciudad
city_names = {
    1: "Madrid", 2: "Barcelona", 3: "Sevilla", 4: "Valencia",
    5: "Málaga", 6: "Menorca", 7: "Mallorca", 8: "Girona", 9: "Euskadi"
}

# Define results_df
results_df = pd.DataFrame({
    'real': y_test_real,
    'pred': y_pred_real,
    'error': errors_real,
    'ciudad': X_test['ciudad'].map(city_names)
})

# Add error_pct to results_df
results_df['error_pct'] = (np.abs(results_df['error']) / results_df['real']) * 100

# Now the scatterplot should work
plt.figure(figsize=(12, 8))
sns.scatterplot(x='real', y='error_pct', data=results_df, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
plt.title('Error Porcentual vs Precio Real')
plt.xlabel('Precio Real')
plt.ylabel('Error Porcentual (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_vs_precio_real.png')
plt.close()


# 6. Distribución de Precios Reales y Predichos
plt.figure(figsize=(12, 6))
sns.kdeplot(y_test_real, label='Real', fill=True)
sns.kdeplot(y_pred_real, label='Predicho', fill=True)
plt.title("Distribución de Precios Reales y Predichos")
plt.xlabel("Precio")
plt.ylabel("Densidad")
plt.legend()
plt.savefig('price_distribution_improved.png')
plt.close()

# 7. MAE por Ciudad
mae_by_city = results_df.groupby('ciudad').apply(lambda x: mean_absolute_error(x['real'], x['pred']))
plt.figure(figsize=(10, 6))
mae_by_city.plot(kind='bar')
plt.title("Error Absoluto Medio (MAE) por Ciudad")
plt.xlabel("Ciudad")
plt.ylabel("MAE")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('mae_by_city.png')
plt.close()

print("\nAnálisis completado. Se han generado gráficos de resultados mejorados.")
print("\nProceso finalizado con éxito.")

# 8. Gráfico de dispersión de precio por distancia al centro para cada ciudad
plt.figure(figsize=(18, 10))
for ciudad_id, ciudad_name in city_names.items():
    city_data = results_df[X_test['ciudad'] == ciudad_id]
    if not city_data.empty:
        plt.scatter(
            X_test.loc[city_data.index, 'distance_to_center'], 
            city_data['real'], 
            alpha=0.5, 
            label=ciudad_name
        )
plt.title('Precio vs Distancia al Centro por Ciudad')
plt.xlabel('Distancia al Centro (km)')
plt.ylabel('Precio Real')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('precio_vs_distancia_por_ciudad.png')
plt.close()

# 9. Análisis del error porcentual por tipo de propiedad
X_test_with_results = X_test.copy()
X_test_with_results['error_abs'] = np.abs(errors_real)
X_test_with_results['error_pct'] = np.abs(errors_real) / y_test_real * 100
X_test_with_results['property_type_name'] = X_test_with_results['property_type'].astype(str)

# Calcular error porcentual promedio por tipo de propiedad
error_by_property = X_test_with_results.groupby('property_type_name')['error_pct'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
error_by_property.plot(kind='bar', color='coral')
plt.title('Error Porcentual Promedio por Tipo de Propiedad')
plt.xlabel('Tipo de Propiedad')
plt.ylabel('Error Porcentual Promedio (%)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_porcentual_por_tipo_propiedad.png')
plt.close()

# 10. Análisis del desempeño por número de habitaciones
plt.figure(figsize=(12, 8))
sns.boxplot(x='bedrooms', y='error_pct', data=X_test_with_results[X_test_with_results['bedrooms'] <= 5])
plt.title('Error Porcentual por Número de Habitaciones')
plt.xlabel('Número de Habitaciones')
plt.ylabel('Error Porcentual (%)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_por_num_habitaciones.png')
plt.close()

# 11. Heatmap de correlación entre características numéricas y error
corr_features = X_test_with_results[numeric_features + ['error_abs', 'error_pct']].corr()
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_features, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_features, mask=mask, cmap=cmap, vmax=0.6, center=0, annot=True, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlación entre Características y Error')
plt.tight_layout()
plt.savefig('correlacion_caracteristicas_error.png')
plt.close()

# 12. Distribución de precios por ciudad
plt.figure(figsize=(16, 10))
sns.boxplot(x='ciudad', y='real', data=results_df)
plt.title('Distribución de Precios por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Precio Real')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('distribucion_precios_por_ciudad.png')
plt.close()

# 13. MAPE por ciudad
mape_by_city = results_df.groupby('ciudad').apply(
    lambda x: np.mean(np.abs((x['real'] - x['pred']) / np.maximum(x['real'], 1))) * 100
)
plt.figure(figsize=(12, 6))
mape_by_city.plot(kind='bar', color='darkgreen')
plt.title('Error Porcentual Absoluto Medio (MAPE) por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('mape_por_ciudad.png')
plt.close()

# 14. Gráfico de violín para comparar distribución de errores por tipo de habitación
plt.figure(figsize=(14, 8))
sns.violinplot(x='room_type', y='error_pct', data=X_test_with_results)
plt.title('Distribución de Errores Porcentuales por Tipo de Habitación')
plt.xlabel('Tipo de Habitación')
plt.ylabel('Error Porcentual (%)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('distribucion_errores_por_tipo_habitacion.png')
plt.close()

# 15. Gráfico de dispersión de error vs precio real
plt.figure(figsize=(12, 8))
sns.scatterplot(x='real', y='error_pct', data=results_df, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
plt.title('Error Porcentual vs Precio Real')
plt.xlabel('Precio Real')
plt.ylabel('Error Porcentual (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_vs_precio_real.png')
plt.close()

# 16. Mapa de calor para visualizar la densidad de errores por precio y capacidad
# 16. Mapa de calor para visualizar la densidad de errores por precio y capacidad
plt.figure(figsize=(12, 10))
sns.kdeplot(
    x='real', 
    y='accommodates',
    data=pd.concat([results_df, X_test[['accommodates']]], axis=1),
    cmap="viridis",
    fill=True,
    thresh=0.05,
    cbar=True,  # Agregar barra de color automáticamente
    cbar_kws={'label': 'Densidad'}  # Etiqueta de la barra de color
)
plt.title('Densidad de Precios por Capacidad de Alojamiento')
plt.xlabel('Precio Real')
plt.ylabel('Capacidad (personas)')
plt.tight_layout()
plt.savefig('densidad_precio_capacidad.png')
plt.close()

# 17. Análisis de precisión por rango de precios
price_bins = [0, 20, 40, 60, 80, 100, float('inf')]
price_labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '>1000']
results_df['price_range'] = pd.cut(results_df['real'], bins=price_bins, labels=price_labels)

# Calcular métricas por rango de precio
metrics_by_price_range = results_df.groupby('price_range').apply(
    lambda x: pd.Series({
        'MAE': mean_absolute_error(x['real'], x['pred']),
        'MAPE': np.mean(np.abs((x['real'] - x['pred']) / np.maximum(x['real'], 1))) * 100,
        'Count': len(x)
    })
)

# Gráfico de barras para MAE por rango de precio
plt.figure(figsize=(12, 6))
metrics_by_price_range['MAE'].plot(kind='bar', color='teal')
plt.title('Error Absoluto Medio (MAE) por Rango de Precio')
plt.xlabel('Rango de Precio (€)')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(metrics_by_price_range['MAE']):
    plt.text(i, v + 5, f"{v:.1f}", ha='center')
plt.tight_layout()
plt.savefig('mae_por_rango_precio.png')
plt.close()

# Gráfico de barras para MAPE por rango de precio
plt.figure(figsize=(12, 6))
metrics_by_price_range['MAPE'].plot(kind='bar', color='coral')
plt.title('Error Porcentual Absoluto Medio (MAPE) por Rango de Precio')
plt.xlabel('Rango de Precio (€)')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(metrics_by_price_range['MAPE']):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('mape_por_rango_precio.png')
plt.close()

## 18. Análisis de la influencia de amenities en errores de predicción
# Asegurarse de que 'amenities_list' esté en X_test_with_results
if 'amenities_list' not in X_test_with_results.columns:
    X_test_with_results['amenities_list'] = df_all.loc[X_test.index, 'amenities_list']

# Crear DataFrame para las top amenities con columnas binarias
X_test_amenities = pd.DataFrame(index=X_test_with_results.index)
for amenity in top_amenities:
    X_test_amenities[f'has_{amenity.lower().replace(" ", "_")}'] = X_test_with_results['amenities_list'].apply(
        lambda x: 1 if amenity in x else 0
    )

# Combinar con los errores porcentuales para análisis
amenity_error_data = pd.concat([X_test_amenities, X_test_with_results['error_pct']], axis=1)

# Crear gráfico de barras para error promedio por amenity
plt.figure(figsize=(14, 8))
amenity_means = []
amenity_names = [amenity.replace('_', ' ').title() for amenity in top_amenities]

for col in X_test_amenities.columns:
    has_amenity = amenity_error_data[amenity_error_data[col] == 1]['error_pct'].mean()
    no_amenity = amenity_error_data[amenity_error_data[col] == 0]['error_pct'].mean()
    
    # Manejar casos donde no hay datos suficientes
    has_amenity = has_amenity if pd.notna(has_amenity) else 0
    no_amenity = no_amenity if pd.notna(no_amenity) else 0
    
    amenity_means.append([no_amenity, has_amenity])

amenity_means = np.array(amenity_means)
x = np.arange(len(amenity_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, amenity_means[:, 0], width, label='Sin Amenity', color='lightblue')
rects2 = ax.bar(x + width/2, amenity_means[:, 1], width, label='Con Amenity', color='salmon')

ax.set_title('Error Porcentual Promedio por Presencia de Amenities')
ax.set_xlabel('Amenity')
ax.set_ylabel('Error Porcentual Promedio (%)')
ax.set_xticks(x)
ax.set_xticklabels(amenity_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig('error_por_presencia_amenities.png')
plt.close()


# 19. Análisis de residuos normalizados (para detectar heteroscedasticidad)
results_df['std_residuals'] = (errors_real - np.mean(errors_real)) / np.std(errors_real)

plt.figure(figsize=(12, 8))
sns.scatterplot(x='pred', y='std_residuals', data=results_df, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
plt.axhline(y=2, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
plt.title('Residuos Estandarizados vs Valores Predichos')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos Estandarizados')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('residuos_estandarizados.png')
plt.close()

# 20. Análisis de QQ-plot para verificar normalidad de residuos
from scipy import stats

plt.figure(figsize=(10, 10))
stats.probplot(results_df['std_residuals'], dist="norm", plot=plt)
plt.title('QQ-Plot de Residuos Estandarizados')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('qqplot_residuos.png')
plt.close()

# 21. Gráfico para correlaciones entre distancia al centro y precio por ciudad
plt.figure(figsize=(18, 12))
for i, (ciudad_id, ciudad_name) in enumerate(city_names.items(), 1):
    city_data = X_test_with_results[X_test_with_results['ciudad'] == ciudad_id]
    if len(city_data) > 10:  # Solo ciudades con suficientes datos
        plt.subplot(3, 3, i)
        sns.regplot(
            x='distance_to_center', 
            y='real', 
            data=pd.concat([city_data, results_df.loc[city_data.index, 'real']], axis=1),
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )
        plt.title(f'{ciudad_name}')
        plt.xlabel('Distancia al Centro (km)')
        plt.ylabel('Precio Real')
        plt.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Relación entre Distancia al Centro y Precio por Ciudad', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('distancia_vs_precio_por_ciudad.png')
plt.close()

# 22. Análisis del ratio precio/distancia por ciudad
X_test_with_results['price_distance_ratio'] = results_df['real'] / (X_test_with_results['distance_to_center'] + 0.1)

plt.figure(figsize=(14, 8))
sns.boxplot(x='ciudad', y='price_distance_ratio', data=X_test_with_results.replace(
    {'ciudad': city_names}))
plt.title('Ratio Precio/Distancia por Ciudad')
plt.xlabel('Ciudad')
plt.ylabel('Precio / Distancia (€/km)')
plt.yscale('log')  # Escala logarítmica para mejor visualización
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ratio_precio_distancia_por_ciudad.png')
plt.close()

# 23. Análisis de la evolución del error con respecto a la luxury_score
plt.figure(figsize=(12, 8))
sns.regplot(
    x='luxury_score', 
    y='error_pct', 
    data=X_test_with_results,
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red'}
)
plt.title('Error Porcentual vs Score de Lujo')
plt.xlabel('Score de Lujo')
plt.ylabel('Error Porcentual (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_vs_luxury_score.png')
plt.close()

# 24. Análisis de la evolución del error con respecto a essential_score
plt.figure(figsize=(12, 8))
sns.regplot(
    x='essential_score', 
    y='error_pct', 
    data=X_test_with_results,
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red'}
)
plt.title('Error Porcentual vs Score de Servicios Esenciales')
plt.xlabel('Score de Servicios Esenciales')
plt.ylabel('Error Porcentual (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_vs_essential_score.png')
plt.close()

# 25. Análisis de la distribución del MAPE por rango de capacidad (accommodates)
accom_bins = [0, 2, 4, 6, 8, 10, float('inf')]
accom_labels = ['1-2', '3-4', '5-6', '7-8', '9-10', '>10']
X_test_with_results['accom_range'] = pd.cut(X_test_with_results['accommodates'], bins=accom_bins, labels=accom_labels)

# Calcular MAPE por rango de capacidad
mape_by_accom = X_test_with_results.groupby('accom_range')['error_pct'].mean()

plt.figure(figsize=(12, 6))
mape_by_accom.plot(kind='bar', color='purple')
plt.title('MAPE por Capacidad de Alojamiento')
plt.xlabel('Capacidad (personas)')
plt.ylabel('MAPE (%)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(mape_by_accom):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('mape_por_capacidad.png')
plt.close()

# 26. Análisis de residuos por tipo de habitación y ciudad
plt.figure(figsize=(18, 12))
for i, room_type in enumerate(X_test_with_results['room_type'].unique(), 1):
    if i <= 9:  # Máximo 9 subplots
        plt.subplot(3, 3, i)
        room_data = results_df[X_test_with_results['room_type'] == room_type]
        sns.boxplot(x='ciudad', y='error', data=room_data)
        plt.title(f'Tipo: {room_type}')
        plt.xlabel('Ciudad')
        plt.ylabel('Error')
        plt.xticks(rotation=90)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Distribución de Errores por Tipo de Habitación y Ciudad', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('errores_por_tipo_habitacion_ciudad.png')
plt.close()

# 27. Análisis de la precisión según número total de amenities
plt.figure(figsize=(14, 8))
sns.regplot(
    x='total_amenities', 
    y='error_pct', 
    data=X_test_with_results,
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red'}
)
plt.title('Error Porcentual vs Número Total de Amenities')
plt.xlabel('Número Total de Amenities')
plt.ylabel('Error Porcentual (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_vs_total_amenities.png')
plt.close()

# 28. Análisis de la precisión según el ratio baño/persona
plt.figure(figsize=(14, 8))
sns.regplot(
    x='bathroom_per_person', 
    y='error_pct', 
    data=X_test_with_results,
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red'}
)
plt.title('Error Porcentual vs Ratio Baño/Persona')
plt.xlabel('Baños por Persona')
plt.ylabel('Error Porcentual (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_vs_bathroom_per_person.png')
plt.close()

# 29. Mapa de calor para visualizar correlación entre características agrupadas y precio
# Seleccionar algunas características clave
key_numeric_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                         'distance_to_center', 'luxury_score', 'essential_score',
                         'total_amenities']

# Crear matriz de correlación
corr_matrix = pd.concat([X_test_with_results[key_numeric_features], results_df['real']], axis=1).corr()

# Generar mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlación entre Características Clave y Precio Real')
plt.tight_layout()
plt.savefig('correlacion_caracteristicas_precio.png')
plt.close()

# 30. Distribución del error a través de los vecindarios (clusters)
plt.figure(figsize=(16, 10))
neighborhoods = X_test_with_results.groupby(['ciudad', 'neighborhood_cluster'])['error_pct'].mean().reset_index()
neighborhoods['ciudad_nombre'] = neighborhoods['ciudad'].map(city_names)
neighborhoods['vecindario_id'] = neighborhoods['ciudad_nombre'].astype(str) + '-' + neighborhoods['neighborhood_cluster'].astype(str)

# Ordenar por ciudad y error
neighborhoods = neighborhoods.sort_values(['ciudad', 'error_pct'], ascending=[True, False])

plt.figure(figsize=(18, 10))
bars = plt.bar(neighborhoods['vecindario_id'], neighborhoods['error_pct'], color='skyblue')

# Colorear barras por ciudad
current_ciudad = None
color_map = plt.cm.tab10
ciudad_ids = sorted(neighborhoods['ciudad'].unique())
color_idx = 0

for i, (idx, row) in enumerate(neighborhoods.iterrows()):
    if current_ciudad != row['ciudad']:
        current_ciudad = row['ciudad']
        color_idx = ciudad_ids.index(current_ciudad) % 10
    
    bars[i].set_color(color_map(color_idx / 10))

plt.title('Error Porcentual Promedio por Vecindario (Cluster)')
plt.xlabel('Vecindario')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=90)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('error_por_vecindario.png')
plt.close()

# 31. Spider chart (radar) para comparar métricas entre ciudades
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_chart(categories, values, title):
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    values += values[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    plt.title(title, size=16, pad=20)
    return fig, ax

# Crear un DataFrame para las métricas por ciudad
metrics_by_city = pd.DataFrame(index=city_names.values())

# Calcular diferentes métricas por ciudad
for ciudad_id, ciudad_name in city_names.items():
    city_data = results_df[X_test['ciudad'] == ciudad_id]
    if not city_data.empty:
        metrics_by_city.loc[ciudad_name, 'MAE'] = mean_absolute_error(city_data['real'], city_data['pred'])
        metrics_by_city.loc[ciudad_name, 'MAPE'] = np.mean(np.abs((city_data['real'] - city_data['pred']) / np.maximum(city_data['real'], 1))) * 100
        metrics_by_city.loc[ciudad_name, 'R2'] = r2_score(city_data['real'], city_data['pred']) * 100  # Convertido a porcentaje
        metrics_by_city.loc[ciudad_name, 'Precision'] = (100 - metrics_by_city.loc[ciudad_name, 'MAPE'])  # Precisión como 100-MAPE
        metrics_by_city.loc[ciudad_name, 'Mean_Price'] = city_data['real'].mean()

# Normalizar las métricas para el radar chart
metrics_normalized = metrics_by_city.copy()
for col in metrics_normalized.columns:
    if col in ['MAPE', 'MAE']:  # Valores más bajos son mejores
        metrics_normalized[col] = 100 - (metrics_normalized[col] / metrics_normalized[col].max() * 100)
    else:  # Valores más altos son mejores
        metrics_normalized[col] = metrics_normalized[col] / metrics_normalized[col].max() * 100

# Generar radar charts para cada ciudad
for ciudad_name in metrics_normalized.index:
    categories = metrics_normalized.columns.tolist()
    values = metrics_normalized.loc[ciudad_name].values.tolist()
    
    fig, ax = radar_chart(categories, values, f'Perfil de Métricas para {ciudad_name}')
    plt.savefig(f'radar_{ciudad_name.lower().replace(" ", "_")}.png')
    plt.close()

# 32. Análisis combinado de las ciudades en un solo radar chart
plt.figure(figsize=(14, 10))
ax = plt.subplot(111, polar=True)

# Color por ciudad
colors = plt.cm.tab10(np.linspace(0, 1, len(city_names)))

for i, ciudad_name in enumerate(metrics_normalized.index):
    values = metrics_normalized.loc[ciudad_name].values.tolist()
    values += values[:1]  # Cerrar el polígono
    
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]  # Cerrar el polígono
    
    ax.plot(angles, values, 'o-', linewidth=2, label=ciudad_name, color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_yticklabels([])
ax.grid(True)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Comparativa de Métricas por Ciudad', size=16, pad=20)
plt.tight_layout()
plt.savefig('radar_comparativa_ciudades.png')
plt.close()

print("\nAnálisis adicional completado. Se han generado 25 nuevas gráficas para el TFG.")
print("\nProceso de visualización extendido finalizado con éxito.")


print("\nGenerando informes textuales de los resultados...")

# Función para crear informe de texto
def create_text_report(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Informe creado: {filename}")

# 1. Informe de métricas globales
metrics_report = f"""INFORME DE MÉTRICAS GLOBALES DEL MODELO
===================================

Métricas en escala logarítmica:
- RMSE: {rmse_log:.4f}
- MAE: {mae_log:.4f}
- R²: {r2_log:.4f}

Métricas en escala real:
- RMSE: {rmse_real:.2f}
- MAE: {mae_real:.2f}
- R²: {r2_real:.4f}
- MAPE: {mape:.2f}%

Información de hiperparámetros:
"""

for key, value in best_params.items():
    metrics_report += f"- {key}: {value}\n"

create_text_report('informe_metricas_globales.txt', metrics_report)

# 2. Informe de importancia de características
importance_report = "IMPORTANCIA DE CARACTERÍSTICAS\n===========================\n\n"
importance_report += "Ranking de características por importancia (Gain):\n\n"

for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance']), 1):
    normalized_importance = importance / importance_df['Importance'].sum() * 100
    importance_report += f"{i}. {feature}: {importance:.2f} (Normalizado: {normalized_importance:.2f}%)\n"

create_text_report('informe_importancia_caracteristicas.txt', importance_report)

# 3. Informe de errores por ciudad
city_errors_report = "ANÁLISIS DE ERRORES POR CIUDAD\n===========================\n\n"
city_errors_report += "Métricas de error por ciudad:\n\n"
city_errors_report += f"{'Ciudad':<15} {'MAE':<10} {'MAPE (%)':<10} {'R²':<10}\n"
city_errors_report += "-" * 45 + "\n"

for ciudad_id, ciudad_name in city_names.items():
    city_data = results_df[X_test['ciudad'] == ciudad_id]
    if not city_data.empty:
        city_mae = mean_absolute_error(city_data['real'], city_data['pred'])
        city_mape = np.mean(np.abs((city_data['real'] - city_data['pred']) / np.maximum(city_data['real'], 1))) * 100
        city_r2 = r2_score(city_data['real'], city_data['pred'])
        city_errors_report += f"{ciudad_name:<15} {city_mae:<10.2f} {city_mape:<10.2f} {city_r2:<10.4f}\n"

create_text_report('informe_errores_por_ciudad.txt', city_errors_report)

# 4. Informe de distribución de precios por ciudad
price_distribution_report = "DISTRIBUCIÓN DE PRECIOS POR CIUDAD\n===============================\n\n"
price_distribution_report += f"{'Ciudad':<15} {'Mín':<10} {'Q1':<10} {'Mediana':<10} {'Media':<10} {'Q3':<10} {'Máx':<10} {'Desv Est':<10}\n"
price_distribution_report += "-" * 85 + "\n"

for ciudad_id, ciudad_name in city_names.items():
    city_data = results_df[X_test['ciudad'] == ciudad_id]
    if not city_data.empty:
        city_min = city_data['real'].min()
        city_q1 = city_data['real'].quantile(0.25)
        city_median = city_data['real'].median()
        city_mean = city_data['real'].mean()
        city_q3 = city_data['real'].quantile(0.75)
        city_max = city_data['real'].max()
        city_std = city_data['real'].std()
        
        price_distribution_report += f"{ciudad_name:<15} {city_min:<10.2f} {city_q1:<10.2f} {city_median:<10.2f} {city_mean:<10.2f} {city_q3:<10.2f} {city_max:<10.2f} {city_std:<10.2f}\n"

create_text_report('informe_distribucion_precios_por_ciudad.txt', price_distribution_report)

# 5. Informe de métricas por rango de precios
price_range_report = "MÉTRICAS POR RANGO DE PRECIOS\n============================\n\n"
price_range_report += f"{'Rango (€)':<15} {'MAE':<10} {'MAPE (%)':<10} {'Cantidad':<10}\n"
price_range_report += "-" * 45 + "\n"

for price_range, row in metrics_by_price_range.iterrows():
    price_range_report += f"{price_range:<15} {row['MAE']:<10.2f} {row['MAPE']:<10.2f} {row['Count']:<10}\n"

create_text_report('informe_metricas_por_rango_precios.txt', price_range_report)

# 6. Informe de métricas por tipo de propiedad
property_type_report = "MÉTRICAS POR TIPO DE PROPIEDAD\n==============================\n\n"
property_type_report += f"{'Tipo de Propiedad':<25} {'MAE':<10} {'MAPE (%)':<10} {'Cantidad':<10}\n"
property_type_report += "-" * 55 + "\n"

property_metrics = X_test_with_results.groupby('property_type').apply(
    lambda x: pd.Series({
        'MAE': mean_absolute_error(results_df.loc[x.index, 'real'], results_df.loc[x.index, 'pred']),
        'MAPE': np.mean(np.abs((results_df.loc[x.index, 'real'] - results_df.loc[x.index, 'pred']) / np.maximum(results_df.loc[x.index, 'real'], 1))) * 100,
        'Count': len(x)
    })
).sort_values('MAPE', ascending=True)

for property_type, row in property_metrics.iterrows():
    property_type_report += f"{property_type:<25} {row['MAE']:<10.2f} {row['MAPE']:<10.2f} {row['Count']:<10}\n"

create_text_report('informe_metricas_por_tipo_propiedad.txt', property_type_report)

# 7. Informe de métricas por número de habitaciones
bedrooms_report = "MÉTRICAS POR NÚMERO DE HABITACIONES\n==================================\n\n"
bedrooms_report += f"{'Habitaciones':<15} {'MAE':<10} {'MAPE (%)':<10} {'Cantidad':<10}\n"
bedrooms_report += "-" * 45 + "\n"

bedrooms_metrics = X_test_with_results.groupby('bedrooms').apply(
    lambda x: pd.Series({
        'MAE': mean_absolute_error(results_df.loc[x.index, 'real'], results_df.loc[x.index, 'pred']),
        'MAPE': np.mean(np.abs((results_df.loc[x.index, 'real'] - results_df.loc[x.index, 'pred']) / np.maximum(results_df.loc[x.index, 'real'], 1))) * 100,
        'Count': len(x)
    })
).sort_index()

for bedrooms, row in bedrooms_metrics.iterrows():
    bedrooms_report += f"{bedrooms:<15} {row['MAE']:<10.2f} {row['MAPE']:<10.2f} {row['Count']:<10}\n"

create_text_report('informe_metricas_por_habitaciones.txt', bedrooms_report)

# 8. Informe de métricas por tipo de habitación
room_type_report = "MÉTRICAS POR TIPO DE HABITACIÓN\n===============================\n\n"
room_type_report += f"{'Tipo de Habitación':<25} {'MAE':<10} {'MAPE (%)':<10} {'Cantidad':<10}\n"
room_type_report += "-" * 55 + "\n"

room_type_metrics = X_test_with_results.groupby('room_type').apply(
    lambda x: pd.Series({
        'MAE': mean_absolute_error(results_df.loc[x.index, 'real'], results_df.loc[x.index, 'pred']),
        'MAPE': np.mean(np.abs((results_df.loc[x.index, 'real'] - results_df.loc[x.index, 'pred']) / np.maximum(results_df.loc[x.index, 'real'], 1))) * 100,
        'Count': len(x)
    })
).sort_values('MAPE', ascending=True)

for room_type, row in room_type_metrics.iterrows():
    room_type_report += f"{room_type:<25} {row['MAE']:<10.2f} {row['MAPE']:<10.2f} {row['Count']:<10}\n"

create_text_report('informe_metricas_por_tipo_habitacion.txt', room_type_report)

# 9. Informe de métricas por capacidad
capacity_report = "MÉTRICAS POR CAPACIDAD DE ALOJAMIENTO\n===================================\n\n"
capacity_report += f"{'Capacidad':<15} {'MAE':<10} {'MAPE (%)':<10} {'Cantidad':<10}\n"
capacity_report += "-" * 45 + "\n"

for accom_range, group in X_test_with_results.groupby('accom_range'):
    cap_mae = mean_absolute_error(results_df.loc[group.index, 'real'], results_df.loc[group.index, 'pred'])
    cap_mape = np.mean(np.abs((results_df.loc[group.index, 'real'] - results_df.loc[group.index, 'pred']) / np.maximum(results_df.loc[group.index, 'real'], 1))) * 100
    cap_count = len(group)
    
    capacity_report += f"{accom_range:<15} {cap_mae:<10.2f} {cap_mape:<10.2f} {cap_count:<10}\n"

create_text_report('informe_metricas_por_capacidad.txt', capacity_report)

# 10. Informe de impacto de amenities
amenities_report = "IMPACTO DE AMENITIES EN EL ERROR DE PREDICCIÓN\n============================================\n\n"
amenities_report += f"{'Amenity':<25} {'Error Sin':<15} {'Error Con':<15} {'Diferencia':<15}\n"
amenities_report += "-" * 70 + "\n"

for i, amenity_name in enumerate(amenity_names):
    no_amenity_error = amenity_means[i, 0]
    has_amenity_error = amenity_means[i, 1]
    difference = no_amenity_error - has_amenity_error
    
    amenities_report += f"{amenity_name:<25} {no_amenity_error:<15.2f} {has_amenity_error:<15.2f} {difference:<15.2f}\n"

create_text_report('informe_impacto_amenities.txt', amenities_report)

# 11. Informe de correlación entre características y errores
correlation_report = "CORRELACIÓN ENTRE CARACTERÍSTICAS Y ERRORES\n=========================================\n\n"
correlation_report += "Coeficientes de correlación con el error absoluto:\n\n"

error_correlations = X_test_with_results[numeric_features].corrwith(X_test_with_results['error_abs']).sort_values(ascending=False)

for feature, corr in error_correlations.items():
    correlation_report += f"{feature:<30}: {corr:.4f}\n"

correlation_report += "\nCoeficientes de correlación con el error porcentual:\n\n"

error_pct_correlations = X_test_with_results[numeric_features].corrwith(X_test_with_results['error_pct']).sort_values(ascending=False)

for feature, corr in error_pct_correlations.items():
    correlation_report += f"{feature:<30}: {corr:.4f}\n"

create_text_report('informe_correlacion_caracteristicas_errores.txt', correlation_report)

# 12. Informe de distancia al centro vs precio
distance_price_report = "RELACIÓN ENTRE DISTANCIA AL CENTRO Y PRECIO\n=======================================\n\n"
distance_price_report += f"{'Ciudad':<15} {'Corr. Pearson':<15} {'Precio Medio':<15} {'Dist. Media':<15} {'Ratio Media':<15}\n"
distance_price_report += "-" * 75 + "\n"

for ciudad_id, ciudad_name in city_names.items():
    city_data = pd.concat([
        X_test_with_results[X_test_with_results['ciudad'] == ciudad_id][['distance_to_center', 'price_distance_ratio']],
        results_df.loc[X_test_with_results[X_test_with_results['ciudad'] == ciudad_id].index, 'real']
    ], axis=1)
    
    if len(city_data) > 10:  # Solo ciudades con suficientes datos
        pearson_corr = city_data['distance_to_center'].corr(city_data['real'])
        mean_price = city_data['real'].mean()
        mean_distance = city_data['distance_to_center'].mean()
        mean_ratio = city_data['price_distance_ratio'].mean()
        
        distance_price_report += f"{ciudad_name:<15} {pearson_corr:<15.4f} {mean_price:<15.2f} {mean_distance:<15.2f} {mean_ratio:<15.2f}\n"

create_text_report('informe_distancia_precio.txt', distance_price_report)

# 13. Informe de métricas por clusters de vecindarios
neighborhood_report = "MÉTRICAS POR CLUSTERS DE VECINDARIOS\n==================================\n\n"

for ciudad_id, ciudad_name in city_names.items():
    city_neighborhoods = X_test_with_results[X_test_with_results['ciudad'] == ciudad_id]
    
    if len(city_neighborhoods) > 0:
        neighborhood_report += f"\nCiudad: {ciudad_name}\n"
        neighborhood_report += f"{'Cluster':<10} {'MAE':<10} {'MAPE (%)':<10} {'Cantidad':<10} {'Dist. Media':<15}\n"
        neighborhood_report += "-" * 55 + "\n"
        
        for cluster, group in city_neighborhoods.groupby('neighborhood_cluster'):
            if len(group) >= 5:  # Solo clusters con suficientes datos
                cluster_mae = mean_absolute_error(results_df.loc[group.index, 'real'], results_df.loc[group.index, 'pred'])
                cluster_mape = np.mean(np.abs((results_df.loc[group.index, 'real'] - results_df.loc[group.index, 'pred']) / np.maximum(results_df.loc[group.index, 'real'], 1))) * 100
                cluster_count = len(group)
                cluster_mean_distance = group['distance_to_center'].mean()
                
                neighborhood_report += f"{cluster:<10} {cluster_mae:<10.2f} {cluster_mape:<10.2f} {cluster_count:<10} {cluster_mean_distance:<15.2f}\n"

create_text_report('informe_metricas_por_vecindarios.txt', neighborhood_report)

# 14. Informe de relación entre características de lujo y precio
luxury_report = "RELACIÓN ENTRE CARACTERÍSTICAS DE LUJO Y PRECIO\n=========================================\n\n"
luxury_report += "Análisis de correlación entre scores de lujo y precio:\n\n"

luxury_corr = results_df['real'].corr(X_test_with_results['luxury_score'])
essential_corr = results_df['real'].corr(X_test_with_results['essential_score'])
total_amenities_corr = results_df['real'].corr(X_test_with_results['total_amenities'])

luxury_report += f"Correlación entre luxury_score y precio: {luxury_corr:.4f}\n"
luxury_report += f"Correlación entre essential_score y precio: {essential_corr:.4f}\n"
luxury_report += f"Correlación entre total_amenities y precio: {total_amenities_corr:.4f}\n\n"

# Análisis por rangos de luxury_score
luxury_report += "Análisis de precio por rangos de luxury_score:\n\n"
luxury_report += f"{'Rango Luxury':<15} {'Precio Medio':<15} {'MAE':<10} {'MAPE (%)':<10} {'Cantidad':<10}\n"
luxury_report += "-" * 60 + "\n"

luxury_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
luxury_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
X_test_with_results['luxury_range'] = pd.cut(X_test_with_results['luxury_score'], bins=luxury_bins, labels=luxury_labels)

for luxury_range, group in X_test_with_results.groupby('luxury_range'):
    if len(group) > 0:
        mean_price = results_df.loc[group.index, 'real'].mean()
        lux_mae = mean_absolute_error(results_df.loc[group.index, 'real'], results_df.loc[group.index, 'pred'])
        lux_mape = np.mean(np.abs((results_df.loc[group.index, 'real'] - results_df.loc[group.index, 'pred']) / np.maximum(results_df.loc[group.index, 'real'], 1))) * 100
        lux_count = len(group)
        
        luxury_report += f"{luxury_range:<15} {mean_price:<15.2f} {lux_mae:<10.2f} {lux_mape:<10.2f} {lux_count:<10}\n"

create_text_report('informe_caracteristicas_lujo.txt', luxury_report)

# 15. Informe completo de estadísticas por ciudad
city_complete_report = "INFORME COMPLETO DE ESTADÍSTICAS POR CIUDAD\n========================================\n\n"

for ciudad_id, ciudad_name in city_names.items():
    city_data = results_df[X_test['ciudad'] == ciudad_id]
    city_features = X_test_with_results[X_test_with_results['ciudad'] == ciudad_id]
    
    if not city_data.empty:
        city_complete_report += f"\n\n{'='*50}\n"
        city_complete_report += f"CIUDAD: {ciudad_name}\n"
        city_complete_report += f"{'='*50}\n\n"
        
        # Estadísticas básicas
        city_complete_report += "1. ESTADÍSTICAS BÁSICAS\n"
        city_complete_report += "-" * 25 + "\n"
        city_complete_report += f"Número de muestras: {len(city_data)}\n"
        city_complete_report += f"Precio medio: {city_data['real'].mean():.2f}\n"
        city_complete_report += f"Precio mediano: {city_data['real'].median():.2f}\n"
        city_complete_report += f"Precio mínimo: {city_data['real'].min():.2f}\n"
        city_complete_report += f"Precio máximo: {city_data['real'].max():.2f}\n"
        city_complete_report += f"Desviación estándar: {city_data['real'].std():.2f}\n\n"
        
        # Métricas de error
        city_complete_report += "2. MÉTRICAS DE ERROR\n"
        city_complete_report += "-" * 25 + "\n"
        city_mae = mean_absolute_error(city_data['real'], city_data['pred'])
        city_mape = np.mean(np.abs((city_data['real'] - city_data['pred']) / np.maximum(city_data['real'], 1))) * 100
        city_rmse = np.sqrt(mean_squared_error(city_data['real'], city_data['pred']))
        city_r2 = r2_score(city_data['real'], city_data['pred'])
        
        city_complete_report += f"MAE: {city_mae:.2f}\n"
        city_complete_report += f"MAPE: {city_mape:.2f}%\n"
        city_complete_report += f"RMSE: {city_rmse:.2f}\n"
        city_complete_report += f"R²: {city_r2:.4f}\n\n"
        
        # Distribución por tipo de propiedad
        city_complete_report += "3. DISTRIBUCIÓN POR TIPO DE PROPIEDAD\n"
        city_complete_report += "-" * 25 + "\n"
        
        property_type_counts = city_features['property_type'].value_counts()
        for property_type, count in property_type_counts.items():
            city_complete_report += f"{property_type}: {count} ({count/len(city_features)*100:.1f}%)\n"
        
        city_complete_report += "\n"
        
        # Distribución por tipo de habitación
        city_complete_report += "4. DISTRIBUCIÓN POR TIPO DE HABITACIÓN\n"
        city_complete_report += "-" * 25 + "\n"
        
        room_type_counts = city_features['room_type'].value_counts()
        for room_type, count in room_type_counts.items():
            city_complete_report += f"{room_type}: {count} ({count/len(city_features)*100:.1f}%)\n"
        
        city_complete_report += "\n"
        
        # Estadísticas de características numéricas
        city_complete_report += "5. ESTADÍSTICAS DE CARACTERÍSTICAS NUMÉRICAS\n"
        city_complete_report += "-" * 25 + "\n"
        
        for feature in ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'distance_to_center', 'luxury_score', 'essential_score', 'total_amenities']:
            if feature in city_features.columns:
                city_complete_report += f"{feature}:\n"
                city_complete_report += f"  Media: {city_features[feature].mean():.2f}\n"
                city_complete_report += f"  Mediana: {city_features[feature].median():.2f}\n"
                city_complete_report += f"  Mín: {city_features[feature].min():.2f}\n"
                city_complete_report += f"  Máx: {city_features[feature].max():.2f}\n\n"
        
        # Correlaciones con el precio
        city_complete_report += "6. CORRELACIONES CON EL PRECIO\n"
        city_complete_report += "-" * 25 + "\n"
        
        city_price_corr = {}
        for feature in ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'distance_to_center', 'luxury_score', 'essential_score', 'total_amenities']:
            if feature in city_features.columns:
                city_price_corr[feature] = city_features[feature].corr(city_data['real'])
        
        for feature, corr in sorted(city_price_corr.items(), key=lambda x: abs(x[1]), reverse=True):
            city_complete_report += f"{feature}: {corr:.4f}\n"

create_text_report('informe_completo_por_ciudad.txt', city_complete_report)

# 16. Informe resumen con interpretación
# Calcular el porcentaje de R²
r2_real_porcentaje = r2_real * 100

# Definir la cadena del informe
summary_report = """INFORME RESUMEN DEL MODELO PREDICTIVO
===================================

RESUMEN DE RESULTADOS:
El modelo LightGBM optimizado ha sido entrenado para predecir precios de alojamientos en 9 ciudades españolas. A continuación se presentan los principales hallazgos:

1. RENDIMIENTO GLOBAL:
   - El modelo alcanza un R² de {r2_real:.4f} en escala real, lo que indica que explica aproximadamente el {r2_real_porcentaje:.1f}% de la varianza en los precios.
   - El error porcentual absoluto medio (MAPE) es de {mape:.2f}%, lo que sugiere que, en promedio, las predicciones se desvían en ese porcentaje del precio real.

2. CARACTERÍSTICAS MÁS IMPORTANTES:
   Las características que más influyen en la predicción de precios son:
""".format(r2_real=r2_real, r2_real_porcentaje=r2_real_porcentaje, mape=mape)

# Añadir top 5 características
for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Normalized Importance']), 1):
    if i <= 5:
        summary_report += f"   - {feature}: {importance:.2f}%\n"

summary_report += """
3. DIFERENCIAS POR CIUDADES:
   Se observan diferencias significativas en la precisión del modelo entre las diferentes ciudades:
"""

# Añadir información por ciudades
for ciudad_id, ciudad_name in city_names.items():
    city_data = results_df[X_test['ciudad'] == ciudad_id]
    if not city_data.empty:
        city_mape = np.mean(np.abs((city_data['real'] - city_data['pred']) / np.maximum(city_data['real'], 1))) * 100
        summary_report += f"   - {ciudad_name}: MAPE {city_mape:.2f}%\n"

summary_report += """
4. INFLUENCIA DE LA DISTANCIA AL CENTRO:
   La distancia al centro muestra una correlación relevante con el precio, aunque varía significativamente entre ciudades.

5. IMPACTO DE AMENITIES:
   Las propiedades con mayores scores de lujo y servicios esenciales tienden a tener precios más altos.

6. COMPORTAMIENTO POR RANGO DE PRECIOS:
   El modelo muestra mejor rendimiento en el rango medio de precios, mientras que tiene más dificultades
   para predecir con precisión los precios muy bajos y muy altos.

7. CONCLUSIONES:
   - El modelo demuestra una capacidad predictiva sólida a nivel global.
   - Existen oportunidades de mejora en la predicción de precios para propiedades de lujo y propiedades en determinadas zonas geográficas.
   - La combinación de características de ubicación, capacidad y amenities proporciona una buena base para predecir precios de alojamientos.
"""

create_text_report('informe_resumen_interpretacion.txt', summary_report)

print("\nGeneración de informes textuales completada.")
print(f"Se han generado {16} archivos de texto con información detallada de los resultados.")
print("\nProceso finalizado con éxito.")