import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from collections import Counter

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
filtered_data["host_age_years"] = (datetime.now() - filtered_data["host_since"]).dt.days / 365  # Antigüedad del host
filtered_data["occupancy_rate"] = (365 - filtered_data["availability_365"]) / 365  # Tasa de ocupación estimada
filtered_data["price_per_person"] = filtered_data["price"] / filtered_data["accommodates"]  # Precio por persona

# Procesar host_response_rate (convertir de porcentaje a decimal si es necesario)
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

# Visualizaciones
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

elif option == "Precios por Vecindario":
    bar_data = filtered_data.groupby("neighbourhood_cleansed")["price"].mean().reset_index()
    fig = px.bar(
        bar_data,
        x="neighbourhood_cleansed",
        y="price",
        title="Precio Promedio por Vecindario",
        color="price",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Cantidad por Tipo de Habitación":
    room_type_counts = filtered_data["room_type"].value_counts().reset_index()
    room_type_counts.columns = ["room_type", "count"]
    fig = px.bar(
        room_type_counts,
        x="room_type",
        y="count",
        title="Cantidad de Alojamientos por Tipo de Habitación",
        color="room_type",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Distribución de Precios":
    fig = px.histogram(
        filtered_data,
        x="price",
        title="Distribución de Precios",
        nbins=50,
        color_discrete_sequence=["#FF6F61"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Relación Precio-Puntuación":
    fig = px.scatter(
        filtered_data,
        x="review_scores_rating",
        y="price",
        title="Relación entre Puntuación de Reseñas y Precio",
        color="room_type",
        hover_data=["neighbourhood_cleansed"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Precio por Tipo de Propiedad":
    bar_data = filtered_data.groupby("property_type")["price"].mean().reset_index()
    fig = px.bar(
        bar_data,
        x="property_type",
        y="price",
        title="Precio Promedio por Tipo de Propiedad",
        color="property_type"
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Precios por Número de Dormitorios":
    fig = px.box(
        filtered_data,
        x="bedrooms",
        y="price",
        title="Distribución de Precios por Número de Dormitorios",
        color="bedrooms"
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Antigüedad del Host vs Precio":
    fig = px.scatter(
        filtered_data,
        x="host_age_years",
        y="price",
        title="Relación entre Antigüedad del Host y Precio",
        color="room_type",
        hover_data=["neighbourhood_cleansed"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Frecuencia de Amenities":
    amenity_counts = {amenity: filtered_data[f"has_{amenity}"].sum() for amenity in common_amenities}
    amenity_df = pd.DataFrame(list(amenity_counts.items()), columns=["amenity", "count"])
    fig = px.bar(
        amenity_df,
        x="amenity",
        y="count",
        title="Frecuencia de las 10 Amenities Más Comunes",
        color="amenity"
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Impacto de Wifi en Precio":
    wifi_data = filtered_data.groupby("has_Wifi")["price"].mean().reset_index()
    fig = px.bar(
        wifi_data,
        x="has_Wifi",
        y="price",
        title="Precio Promedio con y sin Wifi",
        color="has_Wifi"
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Tasa de Respuesta vs Puntuación":
    fig = px.scatter(
        filtered_data,
        x="host_response_rate",
        y="review_scores_communication",
        title="Tasa de Respuesta del Host vs Puntuación de Comunicación",
        color="room_type",
        hover_data=["neighbourhood_cleansed"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Tiempo de Respuesta vs Comunicación":
    bar_data = filtered_data.groupby("host_response_time")["review_scores_communication"].mean().reset_index()
    fig = px.bar(
        bar_data,
        x="host_response_time",
        y="review_scores_communication",
        title="Puntuación de Comunicación por Tiempo de Respuesta",
        color="host_response_time"
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Disponibilidad vs Precio":
    fig = px.scatter(
        filtered_data,
        x="availability_365",
        y="price",
        title="Relación entre Disponibilidad Anual y Precio",
        color="room_type",
        hover_data=["neighbourhood_cleansed"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Capacidad vs Precio por Persona":
    fig = px.scatter(
        filtered_data,
        x="accommodates",
        y="price_per_person",
        title="Capacidad vs Precio por Persona",
        color="room_type",
        hover_data=["neighbourhood_cleansed"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Puntuación de Limpieza vs Precio":
    fig = px.scatter(
        filtered_data,
        x="review_scores_cleanliness",
        y="price",
        title="Puntuación de Limpieza vs Precio",
        color="room_type",
        hover_data=["neighbourhood_cleansed"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Listados del Host vs Precio":
    fig = px.scatter(
        filtered_data,
        x="host_total_listings_count",
        y="price",
        title="Número Total de Listados del Host vs Precio",
        color="room_type",
        hover_data=["neighbourhood_cleansed"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Distribución de Noches Mínimas":
    fig = px.histogram(
        filtered_data,
        x="minimum_nights",
        title="Distribución de Noches Mínimas",
        nbins=50,
        color_discrete_sequence=["#FF6F61"]
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Puntuación de Ubicación vs Precio":
    fig = px.scatter(
        filtered_data,
        x="review_scores_location",
        y="price",
        title="Puntuación de Ubicación vs Precio",
        color="room_type",
        hover_data=["neighbourhood_cleansed"]
    )
    st.plotly_chart(fig, use_container_width=True)

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

# Pie de página personalizado
st.markdown("---")
st.markdown("TFG - Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb | Ángel Soto García")
