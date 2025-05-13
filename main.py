import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from collections import Counter
import joblib
import requests
from io import BytesIO
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Análisis Predictivo de Precios y Reseñas en Airbnb",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ocultar el pie de página "Made with Streamlit"
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Diccionario de ciudades y URLs de GitHub
ciudades_urls = {
    "Barcelona": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_barcelona.parquet",
    "Euskadi": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_euskadi.parquet",
    "Girona": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_girona.parquet",
    "Madrid": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_madrid.parquet",
    "Mallorca": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_mallorca.parquet",
    "Menorca": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_menorca.parquet",
    "Málaga": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_málaga.parquet",
    "Sevilla": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_sevilla.parquet",
    "Valencia": "https://raw.githubusercontent.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/main/datasets/inmuebles_valencia.parquet"
}

# Función para cargar el modelo
@st.cache_resource
def load_model():
    model_url = "https://github.com/asotogarc/TFG-UOC-CienciaDeDatos-062025/raw/main/modelado/PREDICT_RENT_BEST_MODEL.pkl"
    response = requests.get(model_url)
    if response.status_code == 200:
        model = joblib.load(BytesIO(response.content))
        return model
    else:
        st.error("Error al cargar el modelo desde GitHub.")
        return None

# Cargar el modelo
model = load_model()

# Navegación entre páginas
st.sidebar.header("Navegación")
page = st.sidebar.radio("Selecciona una página:", ["Análisis de Datos", "Predicción de Precios"])

if page == "Análisis de Datos":
    # Título principal
    st.title("Análisis de Datos de Airbnb en España 2024")

    # Introducción breve
    st.markdown("""
    Bienvenido al dashboard interactivo para el análisis de datos de Airbnb en diferentes ciudades de España (2024).  
    Este proyecto, parte de mi TFG, explora:  
    - **Predicción de precios** mediante modelos de aprendizaje automático.  
    - **Análisis de reseñas** usando procesamiento de lenguaje natural.  
    Autor: Ángel Soto García - Grado en Ciencia de Datos - UOC
    """)

    # Sidebar para selección de ciudad
    st.sidebar.header("Selección de Ciudad")
    ciudad_seleccionada = st.sidebar.selectbox("Selecciona una ciudad:", list(ciudades_urls.keys()))

    # Cargar datos de la ciudad seleccionada
    try:
        data = pd.read_parquet(ciudades_urls[ciudad_seleccionada])
    except Exception as e:
        st.error(f"Error al cargar los datos de {ciudad_seleccionada}: {e}")
        st.stop()

    # Limpiar datos de vecindarios
    if "neighbourhood_cleansed" in data.columns:
        data["neighbourhood_cleansed"] = data["neighbourhood_cleansed"].astype(str).replace("nan", None)
        neighborhoods_options = [n for n in data["neighbourhood_cleansed"].unique() if n is not None]
    else:
        st.error("La columna 'neighbourhood_cleansed' no está presente en los datos.")
        st.stop()

    # Verificar que la columna 'room_type' existe
    if "room_type" not in data.columns:
        st.error("La columna 'room_type' no está presente en los datos.")
        st.stop()

    # Crear una lista limpia de tipos de habitación
    room_type_options = [str(room) for room in data["room_type"].unique() if pd.notna(room) and room is not None]

    # Sidebar para filtros
    st.sidebar.header("Filtros")
    neighborhoods = st.sidebar.multiselect(
        "Seleccionar vecindarios",
        options=neighborhoods_options,
        default=neighborhoods_options
    )

    # Filtro para room_type
    room_types = st.sidebar.multiselect(
        "Seleccionar tipos de habitación",
        options=room_type_options,
        default=room_type_options
    )

    # Manejar rango de precios
    price_min = float(data["price"].min()) if not data["price"].empty else 0.0
    price_max = float(data["price"].max()) if not data["price"].empty else 1000.0
    price_range = st.sidebar.slider(
        "Rango de precios (€)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max)
    )

    # Filtro para número mínimo de reseñas
    min_reviews = st.sidebar.slider(
        "Número mínimo de reseñas",
        min_value=0,
        max_value=int(data["number_of_reviews"].max()),
        value=0
    )

    # Filtro para noches mínimas
    min_nights_range = st.sidebar.slider(
        "Rango de noches mínimas",
        min_value=int(data["minimum_nights"].min()),
        max_value=int(data["minimum_nights"].max()),
        value=(int(data["minimum_nights"].min()), int(data["minimum_nights"].max()))
    )

    # Filtrar datos según selecciones
    filtered_data = data[
        (data["neighbourhood_cleansed"].isin(neighborhoods)) &
        (data["price"] >= price_range[0]) &
        (data["price"] <= price_range[1]) &
        (data["room_type"].isin(room_types)) &
        (data["number_of_reviews"] >= min_reviews) &
        (data["minimum_nights"] >= min_nights_range[0]) &
        (data["minimum_nights"] <= min_nights_range[1])
    ]

    # Procesar variables y crear nuevas características
    filtered_data["host_since"] = pd.to_datetime(filtered_data["host_since"])
    filtered_data["host_age_years"] = (datetime.now() - filtered_data["host_since"]).dt.days / 365
    filtered_data["occupancy_rate"] = (365 - filtered_data["availability_365"]) / 365
    filtered_data["price_per_person"] = filtered_data["price"] / filtered_data["accommodates"]

    # Procesar host_response_rate
    if filtered_data["host_response_rate"].dtype == object:
        filtered_data["host_response_rate"] = filtered_data["host_response_rate"].str.rstrip("%").astype(float) / 100

    # Extraer las 10 amenities más comunes
    if "amenities" in filtered_data.columns:
        all_amenities = [amenity for sublist in filtered_data["amenities"] for amenity in sublist]
        common_amenities = [item[0] for item in Counter(all_amenities).most_common(10)]
        for amenity in common_amenities:
            filtered_data[f"has_{amenity}"] = filtered_data["amenities"].apply(lambda x: amenity in x)

    # Sección de visualizaciones interactivas
    st.header(f"Visualizaciones para {ciudad_seleccionada}")
    option = st.selectbox(
        "Selecciona el tipo de visualización:",
        [
            "Mapa",
            "Precios por Vecindario",
            "Cantidad por Tipo de Habitación",
            "Distribución de Precios",
            "Relación Precio-Puntuación",
            "Precio por Tipo de Propiedad",
            "Precios por Número de Dormitorios",
            "Antigüedad del Host vs Precio",
            "Frecuencia de Amenities",
            "Impacto de Wifi en Precio",
            "Tasa de Respuesta vs Puntuación",
            "Tiempo de Respuesta vs Comunicación",
            "Disponibilidad vs Precio",
            "Capacidad vs Precio por Persona",
            "Puntuación de Limpieza vs Precio",
            "Listados del Host vs Precio",
            "Distribución de Noches Mínimas",
            "Puntuación de Ubicación vs Precio"
        ]
    )

    # Visualizaciones (mismas que en el código original, abreviadas por brevedad)
    if option == "Mapa":
        fig = px.scatter_mapbox(
            filtered_data,
            lat="latitude",
            lon="longitude",
            color="price",
            size="number_of_reviews",
            hover_name="neighbourhood_cleansed",
            zoom=10,
            title="Distribución Geográfica de Alojamientos",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    # ... (Otras visualizaciones similares al código original)

    # Métricas resumidas
    st.header("Métricas Resumidas")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Precio Promedio", f"€{filtered_data['price'].mean():.2f}")
    with col2:
        st.metric("Número de Alojamientos", len(filtered_data))
    with col3:
        st.metric("Puntuación Promedio", f"{filtered_data['review_scores_rating'].mean():.2f}")
    with col4:
        st.metric("Tasa de Ocupación Promedio", f"{filtered_data['occupancy_rate'].mean():.2%}")
    with col5:
        st.metric("Antigüedad Promedio del Host (años)", f"{filtered_data['host_age_years'].mean():.2f}")

elif page == "Predicción de Precios":
    st.title("Predicción de Precios de Alojamientos en Airbnb")
    st.markdown("""
    Esta página permite predecir el precio de un alojamiento en Airbnb utilizando un modelo LightGBM optimizado.  
    Introduce los parámetros del alojamiento y obtén una predicción del precio por noche.
    """)

    # Formulario para predicción
    st.header("Formulario de Predicción")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            ciudad = st.selectbox("Ciudad", list(ciudades_urls.keys()))
            property_type = st.selectbox("Tipo de Propiedad", ["Apartment", "House", "Condominium", "Loft", "Other"])
            room_type = st.selectbox("Tipo de Habitación", ["Entire home/apt", "Private room", "Shared room"])
            accommodates = st.slider("Capacidad (Personas)", 1, 16, 2)
            bathrooms = st.slider("Baños", 0.0, 8.0, 1.0, step=0.5)
            bedrooms = st.slider("Dormitorios", 0, 10, 1)
            beds = st.slider("Camas", 0, 16, 1)
        with col2:
            minimum_nights = st.slider("Noches Mínimas", 1, 30, 1)
            maximum_nights = st.slider("Noches Máximas", 1, 1125, 1125)
            total_amenities = st.slider("Número Total de Amenities", 0, 50, 10)
            distance_to_center = st.slider("Distancia al Centro (km)", 0.0, 50.0, 1.0)
            neighborhood_cluster = st.slider("Cluster de Vecindario", 0, 19, 0)
            essential_score = st.slider("Puntuación de Amenities Esenciales", 0.0, 5.0, 3.0)
            luxury_score = st.slider("Puntuación de Amenities de Lujo", 0.0, 5.0, 1.0)

        submitted = st.form_submit_button("Predecir Precio")
        if submitted and model is not None:
            # Preparar datos para predicción
            input_data = pd.DataFrame({
                "ciudad": [ciudad],
                "property_type": [property_type],
                "room_type": [room_type],
                "accommodates": [accommodates],
                "bathrooms": [bathrooms],
                "bedrooms": [bedrooms],
                "beds": [beds],
                "minimum_nights": [minimum_nights],
                "maximum_nights": [maximum_nights],
                "total_amenities": [total_amenities],
                "distance_to_center": [distance_to_center],
                "neighborhood_cluster": [neighborhood_cluster],
                "essential_score": [essential_score],
                "luxury_score": [luxury_score],
                "scaled_log_distance": [np.log1p(distance_to_center)],
                "log_accommodates": [np.log1p(accommodates)],
                "scaled_distance_to_center": [distance_to_center / 50.0],
                "scaled_total_amenities": [total_amenities / 50.0],
                "log_distance": [np.log1p(distance_to_center)],
                "log_minimum_nights": [np.log1p(minimum_nights)],
                "log_maximum_nights": [np.log1p(maximum_nights)],
                "bathroom_per_person": [bathrooms / accommodates],
                "accommodates_squared": [accommodates ** 2],
                "bed_bath_product": [beds * bathrooms],
                "bed_to_bedroom_ratio": [beds / max(bedrooms, 1)],
                "person_per_bed": [accommodates / max(beds, 1)],
                "person_per_bedroom": [accommodates / max(bedrooms, 1)],
                "bed_accom_ratio": [beds / accommodates],
                "distance_to_center_squared": [distance_to_center ** 2],
                "bedrooms_squared": [bedrooms ** 2],
                "bathrooms_squared": [bathrooms ** 2],
                "beds_squared": [beds ** 2]
            })

            # Realizar predicción
            try:
                prediction = model.predict(input_data)[0]
                st.success(f"**Precio Predicho:** €{prediction:.2f} por noche")
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

    # Informe del modelo
    st.header("Informe Resumen del Modelo Predictivo")
    st.markdown("""
    ### RESUMEN DE RESULTADOS
    El modelo LightGBM optimizado ha sido entrenado para predecir precios de alojamientos en 9 ciudades españolas. A continuación, se presentan los principales hallazgos:

    1. **RENDIMIENTO GLOBAL**:
       - **R²**: 0.5938 (explica ~59.4% de la varianza en los precios).
       - **MAPE**: 19.22% (error porcentual absoluto medio).

    2. **CARACTERÍSTICAS MÁS IMPORTANTES**:
       - `property_type`: 28.96%
       - `room_type`: 12.52%
       - `accommodates`: 6.83%
       - `distance_to_center`: 6.76%
       - `total_amenities`: 6.59%

    3. **DIFERENCIAS POR CIUDADES**:
       - Madrid: MAPE 20.96%
       - Barcelona: MAPE 24.54%
       - Sevilla: MAPE 16.42%
       - Valencia: MAPE 16.62%
       - Málaga: MAPE 15.20%
       - Menorca: MAPE 15.68%
       - Mallorca: MAPE 17.87%
       - Girona: MAPE 15.25%
       - Euskadi: MAPE 15.61%

    4. **INFLUENCIA DE LA DISTANCIA AL CENTRO**:
       La distancia al centro muestra una correlación relevante con el precio, aunque varía entre ciudades.

    5. **IMPACTO DE AMENITIES**:
       Propiedades con mayores scores de lujo y servicios esenciales tienden a tener precios más altos.

    6. **COMPORTAMIENTO POR RANGO DE PRECIOS**:
       Mejor rendimiento en el rango medio de precios; dificultades en precios muy bajos y muy altos.

    7. **CONCLUSIONES**:
       - Capacidad predictiva sólida a nivel global.
       - Oportunidades de mejora en propiedades de lujo y ciertas zonas geográficas.
       - Ubicación, capacidad y amenities son clave para predecir precios.
    """)

    # Visualización de métricas por ciudad
    st.subheader("Métricas por Ciudad")
    metrics_data = pd.Datahuis({
        "Ciudad": ["Madrid", "Barcelona", "Sevilla", "Valencia", "Málaga", "Menorca", "Mallorca", "Girona", "Euskadi"],
        "MAPE (%)": [20.96, 24.54, 16.42, 16.62, 15.20, 15.68, 17.87, 15.25, 15.61]
    })
    fig_metrics = px.bar(
        metrics_data,
        x="Ciudad",
        y="MAPE (%)",
        title="Error Porcentual Absoluto Medio (MAPE) por Ciudad",
        color="Ciudad",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Visualización de importancia de características
    st.subheader("Importancia de Características")
    features_data = pd.DataFrame({
        "Característica": [
            "property_type", "room_type", "accommodates", "distance_to_center", "total_amenities",
            "scaled_log_distance", "minimum_nights", "maximum_nights", "neighborhood_cluster", "ciudad"
        ],
        "Importancia (%)": [28.96, 12.52, 6.83, 6.76, 6.59, 5.55, 4.90, 3.55, 3.15, 2.37]
    })
    fig_features = px.bar(
        features_data,
        x="Importancia (%)",
        y="Característica",
        orientation="h",
        title="Top 10 Características por Importancia",
        color="Característica",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    st.plotly_chart(fig_features, use_container_width=True)

    # Métricas por clústeres (selección de ciudad)
    st.subheader("Métricas por Clústeres de Vecindarios")
    cluster_city = st.selectbox("Selecciona una ciudad para ver métricas por clúster:", list(ciudades_urls.keys()))
    cluster_data = {
        "Madrid": [
            {"Cluster": i, "MAE": mae, "MAPE (%)": mape, "Cantidad": qty, "Dist. Media": dist}
            for i, (mae, mape, qty, dist) in enumerate([
                (11.29, 19.86, 858, 1.32), (12.02, 25.00, 218, 7.34), (12.51, 24.36, 314, 2.53),
                (10.39, 19.59, 470, 4.54), (11.73, 22.01, 434, 3.84), (10.23, 23.80, 476, 4.63),
                (9.00, 17.28, 447, 3.36), (11.76, 24.03, 193, 5.55), (11.54, 21.60, 330, 5.85),
                (11.68, 20.08, 713, 1.12), (11.88, 23.06, 491, 2.17), (11.49, 20.26, 152, 7.96),
                (10.59, 23.21, 167, 6.89), (11.69, 18.92, 584, 0.44), (10.57, 19.70, 107, 8.38),
                (11.87, 20.92, 1079, 0.88), (9.91, 19.50, 207, 5.44), (11.01, 21.80, 451, 3.53),
                (9.71, 19.82, 91, 9.23), (9.31, 16.38, 128, 11.28)
            ])
        ],
        "Barcelona": [
            {"Cluster": i, "MAE": mae, "MAPE (%)": mape, "Cantidad": qty, "Dist. Media": dist}
            for i, (mae, mape, qty, dist) in enumerate([
                (13.75, 24.51, 537, 0.96), (12.05, 25.81, 238, 2.23), (12.32, 23.12, 214, 3.83),
                (13.55, 24.45, 354, 1.57), (14.03, 26.40, 300, 3.75), (12.86, 24.98, 331, 1.79),
                (11.17, 21.95, 361, 2.30), (11.50, 23.26, 222, 2.91), (12.27, 25.85, 262, 4.85),
                (11.13, 23.55, 247, 0.86), (12.71, 24.05, 236, 3.03), (12.67, 21.46, 54, 5.15),
                (13.54, 25.65, 246, 0.87), (12.37, 25.69, 115, 2.83), (13.21, 23.80, 378, 1.56),
                (11.55, 24.46, 104, 4.39), (13.40, 23.34, 230, 1.84), (14.00, 26.56, 535, 0.79),
                (12.42, 24.33, 292, 1.10), (12.47, 24.04, 181, 1.89)
            ])
        ],
        # ... (Datos similares para Sevilla, Valencia, Málaga, Menorca, Mallorca, Girona, Euskadi)
    }
    if cluster_city in cluster_data:
        cluster_df = pd.DataFrame(cluster_data[cluster_city])
        fig_cluster = px.scatter(
            cluster_df,
            x="Dist. Media",
            y="MAPE (%)",
            size="Cantidad",
            color="Cluster",
            hover_data=["MAE"],
            title=f"MAPE por Clúster en {cluster_city}",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        st.dataframe(cluster_df)

# Pie de página personalizado
st.markdown("---")
st.markdown("TFG - Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb | Ángel Soto García")
