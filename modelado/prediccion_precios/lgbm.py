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

ciudad_centros = {
    1: (40.4168, -3.7038), 2: (41.3879, 2.1700), 3: (37.3772, -5.9869),
    4: (39.4699, -0.3763), 5: (36.7202, -4.4214), 6: (39.8885, 4.2658),
    7: (39.5696, 2.6501), 8: (41.9842, 2.8214), 9: (43.2627, -2.9253)
}

def load_parquet(file_name):
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    file_path = project_root / 'datasets' / file_name
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error al cargar {file_name}: {e}")
        return pd.DataFrame()

df_madrid = load_parquet('inmuebles_madrid.parquet')
df_barcelona = load_parquet('inmuebles_barcelona.parquet')
df_euskadi = load_parquet('inmuebles_euskadi.parquet')
df_sevilla = load_parquet('inmuebles_sevilla.parquet')
df_valencia = load_parquet('inmuebles_valencia.parquet')
df_malaga = load_parquet('inmuebles_málaga.parquet')
df_menorca = load_parquet('inmuebles_menorca.parquet')
df_mallorca = load_parquet('inmuebles_mallorca.parquet')
df_girona = load_parquet('inmuebles_girona.parquet')

df_all = pd.concat([
    df_madrid.assign(ciudad=1), df_barcelona.assign(ciudad=2),
    df_sevilla.assign(ciudad=3), df_valencia.assign(ciudad=4),
    df_malaga.assign(ciudad=5), df_menorca.assign(ciudad=6),
    df_mallorca.assign(ciudad=7), df_girona.assign(ciudad=8),
    df_euskadi.assign(ciudad=9)
]).reset_index(drop=True)

print(f"Tamaño total del conjunto de datos: {df_all.shape}")

for col in ['bathrooms', 'bedrooms', 'beds', 'accommodates']:
    if df_all[col].isnull().any():
        df_all[col].fillna(df_all[col].median(), inplace=True)

df_all = df_all[df_all['price'] > 0]

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

df_all['bathroom_per_person'] = df_all['bathrooms'] / df_all['accommodates'].replace(0, 1)
df_all['bed_to_bedroom_ratio'] = df_all['beds'] / df_all['bedrooms'].replace(0, 1)
df_all['person_per_bedroom'] = df_all['accommodates'] / df_all['bedrooms'].replace(0, 1)
df_all['person_per_bed'] = df_all['accommodates'] / df_all['beds'].replace(0, 1)
df_all['price_per_person'] = df_all['price'] / df_all['accommodates'].replace(0, 1)
df_all['price_per_bedroom'] = df_all['price'] / df_all['bedrooms'].replace(0, 1)
df_all['bed_bath_product'] = df_all['beds'] * df_all['bathrooms']
df_all['bed_accom_ratio'] = df_all['beds'] / df_all['accommodates'].replace(0, 1)

for col in ['minimum_nights', 'maximum_nights', 'accommodates']:
    df_all[f'log_{col}'] = np.log1p(df_all[col])

df_all['amenities_list'] = df_all['amenities'].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) and x != '[]' else []
)
df_all['total_amenities'] = df_all['amenities_list'].apply(len)

all_amenities = [amenity for sublist in df_all['amenities_list'] for amenity in sublist]
top_amenities = [amenity for amenity, count in Counter(all_amenities).most_common(30) 
                 if count > len(df_all) * 0.05]
for amenity in top_amenities:
    df_all[f'has_{amenity}'] = df_all['amenities_list'].apply(lambda x: 1 if amenity in x else 0)

luxury_amenities = ['Pool', 'Hot tub', 'Gym', 'Doorman', 'Elevator']
essential_amenities = ['Wifi', 'Kitchen', 'Heating', 'Air conditioning', 'Washer']
df_all['luxury_score'] = df_all['amenities_list'].apply(
    lambda x: sum(1 for item in luxury_amenities if item in x) / len(luxury_amenities)
)
df_all['essential_score'] = df_all['amenities_list'].apply(
    lambda x: sum(1 for item in essential_amenities if item in x) / len(essential_amenities)
)

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

numeric_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 
    'minimum_nights', 'maximum_nights', 'bathroom_per_person', 
    'bed_to_bedroom_ratio', 'person_per_bedroom', 'person_per_bed',
    'price_per_person', 'price_per_bedroom', 'distance_to_center', 
    'log_distance', 'total_amenities', 'luxury_score', 'essential_score',
    'bed_bath_product', 'bed_accom_ratio', 'log_accommodates', 
    'log_minimum_nights', 'log_maximum_nights'
] + [f'has_{amenity}' for amenity in top_amenities]
categorical_features = ['property_type', 'room_type', 'ciudad', 'neighborhood_cluster']

all_features = numeric_features + categorical_features
missing_cols = [col for col in all_features if col not in df_all.columns]
if missing_cols:
    print(f"Columnas faltantes: {missing_cols}")
    numeric_features = [f for f in numeric_features if f in df_all.columns]
    categorical_features = [f for f in categorical_features if f in df_all.columns]

X = df_all[numeric_features + categorical_features].copy()
y = np.log1p(df_all['price'])

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=df_all['ciudad']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=X_train_full['ciudad']
)
print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

for col in categorical_features:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
    X_test[col] = X_test[col].astype('category')

for col in numeric_features:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
    X_val[col] = pd.to_numeric(X_val[col], errors='coerce').fillna(0)
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

scaler = StandardScaler()
cols_to_scale = ['distance_to_center', 'log_distance', 'total_amenities']
for col in cols_to_scale:
    if col in X_train.columns:
        X_train[f'scaled_{col}'] = scaler.fit_transform(X_train[[col]])
        X_val[f'scaled_{col}'] = scaler.transform(X_val[[col]])
        X_test[f'scaled_{col}'] = scaler.transform(X_test[[col]])
        numeric_features.append(f'scaled_{col}')

key_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'distance_to_center']
for var in key_features:
    if var in X_train.columns:
        X_train[f'{var}_squared'] = X_train[var] ** 2
        X_val[f'{var}_squared'] = X_val[var] ** 2
        X_test[f'{var}_squared'] = X_test[var] ** 2
        numeric_features.append(f'{var}_squared')

train_data = lgb.Dataset(
    X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False
)
valid_data = lgb.Dataset(
    X_val, label=y_val, categorical_feature=categorical_features, reference=train_data, free_raw_data=False
)

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
study.optimize(objective, n_trials=100)
best_params = study.best_params
best_trial = study.best_trial
print("\nMejores parámetros:")
for key, value in best_params.items():
    print(f"    {key}: {value}")
print(f"Mejor MSE: {best_trial.user_attrs['mse']:.6f}")
print(f"Mejor MAPE: {best_trial.user_attrs['mape']:.2f}%")
print(f"Mejor R2: {best_trial.user_attrs['r2']:.6f}")
print(f"Índice de sobreajuste: {best_trial.user_attrs['overfitting']:.6f}")

print("\nEntrenando modelo final con los mejores parámetros...")
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

for col in categorical_features:
    X_train_full[col] = X_train_full[col].astype('category')

train_full_data = lgb.Dataset(
    X_train_full, label=y_train_full, categorical_feature=categorical_features, free_raw_data=False
)

final_params = best_params.copy()
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

model_path = 'lgbm_best_model_improved.txt'
lgbm_best.save_model(model_path)
print(f"Modelo guardado en {model_path}")

print("\nEvaluando modelo en conjunto de prueba...")
y_test_pred = lgbm_best.predict(X_test)

rmse_log = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_log = mean_absolute_error(y_test, y_test_pred)
r2_log = r2_score(y_test, y_test_pred)
print(f"Métricas (escala logarítmica):")
print(f"    RMSE: {rmse_log:.4f}")
print(f"    MAE: {mae_log:.4f}")
print(f"    R²: {r2_log:.4f}")

y_test_real = np.expm1(y_test)
print(y_test_real)
y_pred_real = np.expm1(y_test_pred)
print(y_pred_real)
rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae_real = mean_absolute_error(y_test_real, y_pred_real)
r2_real = r2_score(y_test_real, y_pred_real)
mape = np.mean(np.abs((y_test_real - y_pred_real) / np.maximum(y_test_real, 1))) * 100
print(f"\nMétricas (escala real):")
print(f"    RMSE: {rmse_real:.2f}")
print(f"    MAE: {mae_real:.2f}")
print(f"    R²: {r2_real:.4f}")
print(f"    MAPE: {mape:.2f}%")

print("\nAnalizando importancia de características...")
importance = lgbm_best.feature_importance(importance_type='gain')
feature_names = lgbm_best.feature_name()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)
print("\nTop 30 características más importantes:")
print(importance_df.head(30))

errors_real = y_test_real - y_pred_real
plt.figure(figsize=(12, 6))
sns.histplot(errors_real, kde=True, bins=50, color="blue")
plt.title("Distribución de Errores (Escala Real)")
plt.xlabel("Error en precio real")
plt.ylabel("Frecuencia")
plt.axvline(x=0, linestyle='--', color='red')
plt.savefig('error_distribution.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test_real, y=y_pred_real, alpha=0.4, color="green")
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--')
plt.title("Valores Reales vs Predichos (Escala Real)")
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.savefig('predicted_vs_actual.png')
plt.close()

plt.figure(figsize=(14, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title("Top 20 Características por Importancia (Gain)")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.regplot(x=y_pred_real, y=errors_real, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.axhline(y=0, linestyle='--', color='red')
plt.title("Análisis de Residuos")
plt.xlabel("Valores Predichos")
plt.ylabel("Residuos")
plt.savefig('residuals_analysis.png')
plt.close()

city_names = {
    1: "Madrid", 2: "Barcelona", 3: "Sevilla", 4: "Valencia",
    5: "Málaga", 6: "Menorca", 7: "Mallorca", 8: "Girona", 9: "Euskadi"
}
results_df = pd.DataFrame({
    'real': y_test_real,
    'pred': y_pred_real,
    'error': errors_real,
    'ciudad': X_test['ciudad'].map(city_names)
})
plt.figure(figsize=(14, 6))
sns.boxplot(x='ciudad', y='error', data=results_df)
plt.title("Distribución de Errores por Ciudad")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('errors_by_city.png')
plt.close()

print("\nAnálisis completado. Se han generado gráficos de resultados.")
print("\nProceso finalizado con éxito.")
