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
    default=neighborhoods_options  # Todos los vecindarios válidos por defecto
)

# Filtro para room_type
room_types = st.sidebar.multiselect(
    "Seleccionar tipos de habitación",
    options=room_type_options,
    default=room_type_options  # Todos los tipos por defecto
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
    (data["price"] <= price_range[1]) &
    (data["room_type"].isin(room_types))
]

# Sección de visualizaciones interactivas
st.header(f"Visualizaciones para {ciudad_seleccionada}")
option = st.selectbox(
    "Selecciona el tipo de visualización:",
    ["Mapa", "Precios por Vecindario", "Cantidad por Tipo de Habitación", "Distribución de Precios"]
)

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
    
    # Estadísticas descriptivas
    if not filtered_data.empty:
        st.subheader("Estadísticas Descriptivas por Tipo de Habitación")
        for room_type in room_types:
            room_data = filtered_data[filtered_data["room_type"] == room_type]
            if not room_data.empty:
                st.write(f"**{room_type}**")
                st.write(f"- Número de alojamientos: {len(room_data)}")
                st.write(f"- Precio promedio: €{room_data['price'].mean():.2f}")
                st.write(f"- Puntuación promedio: {room_data['review_scores_rating'].mean():.2f}")
            else:
                st.write(f"No hay datos para {room_type}")
    else:
        st.write("No hay datos para mostrar.")

elif option == "Distribución de Precios":
    fig = px.histogram(
        filtered_data,
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

# Pie de página personalizado
st.markdown("---")
st.markdown("TFG - Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb | Ángel Soto García")
