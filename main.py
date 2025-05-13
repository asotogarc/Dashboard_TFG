import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Análisis Predictivo de Precios y Reseñas en Airbnb",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # Convertir a string y eliminar NaN
    data["neighbourhood_cleansed"] = data["neighbourhood_cleansed"].astype(str).replace("nan", None)
    neighborhoods_options = [n for n in data["neighbourhood_cleansed"].unique() if n is not None]
else:
    st.error("La columna 'neighbourhood_cleansed' no está presente en los datos.")
    st.stop()

# Sidebar para filtros
st.sidebar.header("Filtros")
neighborhoods = st.sidebar.multiselect(
    "Seleccionar vecindarios",
    options=neighborhoods_options,
    default=neighborhoods_options  # Todos los vecindarios válidos por defecto
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

# Filtrar datos según selecciones
filtered_data = data[
    (data["neighbourhood_cleansed"].isin(neighborhoods)) &
    (data["price"] >= price_range[0]) &
    (data["price"] <= price_range[1])
]

# Sección de visualizaciones interactivas
st.header(f"Visualizaciones para {ciudad_seleccionada}")
option = st.selectbox(
    "Selecciona el tipo de visualización:",
    ["Mapa", "Precios por Vecindario", "Precio vs. Puntuación", "Distribución de Precios"]
)

if option == "Mapa":
    fig = px.scatter_mapbox(
        filtered_data,
        lat="latitude",
        lon="longitude",
        color="price",
        size="number_of_reviews",
        hover_name="name",
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

elif option == "Precio vs. Puntuación":
    fig = px.scatter(
        filtered_data,
        x="review_scores_rating",
        y="price",
        color="room_type",
        title="Precio vs. Puntuación por Tipo de Habitación",
        hover_data=["neighbourhood_cleansed"],
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

elif option == "Distribución de Precios":
    fig = px.histogram(
        filtered_data,
K
        x="price",
        title="Distribución de Precios",
        nbins=50,
        color_discrete_sequence=["#FF6F61"]
    )
    st.plotly_chart(fig, use_container_width=True)

# Métricas resumidas
st.header("Métricas Resumidas")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Precio Promedio", f"€{filtered_data['price'].mean():.2f}")
with col2:
    st.metric("Número de Alojamientos", len(filtered_data))
with col3:
    st.metric("Puntuación Promedio", f"{filtered_data['review_scores_rating'].mean():.2f}")

# Pie de página
st.markdown("---")
st.markdown("TFG - Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb | Ángel Soto García")
