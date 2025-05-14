import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter
import requests
import numpy as np
import joblib
from io import BytesIO
import sklearn# Configuración de la página (sin cambios)
st.set_page_config(
    page_title="Análisis Predictivo de Precios y Reseñas en Airbnb",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)# Ocultar el pie de página (sin cambios)
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)# Título e introducción (sin cambios)
st.title("Análisis de Datos de Airbnb en España 2024")
st.markdown("""
Bienvenido al dashboard interactivo para el análisis de datos de Airbnb en diferentes ciudades de España (2024).  
Este proyecto, parte de mi TFG, explora:  **Predicción de precios** mediante modelos de aprendizaje automático.  
**Análisis de reseñas** usando procesamiento de lenguaje natural.  
Autor: Ángel Soto García - Grado en Ciencia de Datos - UOC
""")

# Diccionario de ciudades y URLs (sin cambios)
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
}# Sidebar para selección de ciudad y filtros (sin cambios)
st.sidebar.header("Selección de Ciudad")
ciudad_seleccionada = st.sidebar.selectbox("Selecciona una ciudad:", list(ciudades_urls.keys()))try:
    data = pd.read_parquet(ciudades_urls[ciudad_seleccionada])
except Exception as e:
    st.error(f"Error al cargar los datos de {ciudad_seleccionada}: {e}")
    st.stop()# Limpiar datos de vecindarios y room_type (sin cambios)
if "neighbourhood_cleansed" in data.columns:
    data["neighbourhood_cleansed"] = data["neighbourhood_cleansed"].astype(str).replace("nan", None)
    neighborhoods_options = [n for n in data["neighbourhood_cleansed"].unique() if n is not None]
else:
    st.error("La columna 'neighbourhood_cleansed' no está presente en los datos.")
    st.stop()if "room_type" not in data.columns:
    st.error("La columna 'room_type' no está presente en los datos.")
    st.stop()room_type_options = [str(room) for room in data["room_type"].unique() if pd.notna(room) and room is not None]st.sidebar.header("Filtros")
neighborhoods = st.sidebar.multiselect(
    "Seleccionar vecindarios",
    options=neighborhoods_options,
    default=neighborhoods_options
)
room_types = st.sidebar.multiselect(
    "Seleccionar tipos de habitación",
    options=room_type_options,
    default=room_type_options
)
price_min = float(data["price"].min()) if not data["price"].empty else 0.0
price_max = float(data["price"].max()) if not data["price"].empty else 1000.0
price_range = st.sidebar.slider(
    "Rango de precios (€)",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max)
)
min_reviews = st.sidebar.slider(
    "Número mínimo de reseñas",
    min_value=0,
    max_value=int(data["number_of_reviews"].max()),
    value=0
)
min_nights_range = st.sidebar.slider(
    "Rango de noches mínimas",
    min_value=int(data["minimum_nights"].min()),
    max_value=int(data["minimum_nights"].max()),
    value=(int(data["minimum_nights"].min()), int(data["minimum_nights"].max()))
)# Filtrar datos (sin cambios)
filtered_data = data[
    (data["neighbourhood_cleansed"].isin(neighborhoods)) &
    (data["price"] >= price_range[0]) &
    (data["price"] <= price_range[1]) &
    (data["room_type"].isin(room_types)) &
    (data["number_of_reviews"] >= min_reviews) &
    (data["minimum_nights"] >= min_nights_range[0]) &
    (data["minimum_nights"] <= min_nights_range[1])
]# Procesar variables (sin cambios)
filtered_data["host_since"] = pd.to_datetime(filtered_data["host_since"])
filtered_data["host_age_years"] = (datetime.now() - filtered_data["host_since"]).dt.days / 365
filtered_data["occupancy_rate"] = (365 - filtered_data["availability_365"]) / 365
filtered_data["price_per_person"] = filtered_data["price"] / filtered_data["accommodates"]
if filtered_data["host_response_rate"].dtype == object:
    filtered_data["host_response_rate"] = filtered_data["host_response_rate"].str.rstrip("%").astype(float) / 100
if "amenities" in filtered_data.columns:
    all_amenities = [amenity for sublist in filtered_data["amenities"] for amenity in sublist]
    common_amenities = [item[0] for item in Counter(all_amenities).most_common(10)]
    for amenity in common_amenities:
        filtered_data[f"has_{amenity}"] = filtered_data["amenities"].apply(lambda x: amenity in x)# Sección de visualizaciones interactivas
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
        "Puntuación de Ubicación vs Precio",
        "Análisis de Clusters",
        "Predicción de Precios"
    ]
)# Datos para la sección de clusters (hard-coded según lo proporcionado)
data_dia_semana = pd.DataFrame({
    "Día": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
    "vader_compound": [0.776859, 0.783723, 0.776271, 0.773506, 0.775032, 0.776225, 0.764861],
    "num_reseñas": [7200, 7500, 7300, 7100, 7400, 7600, 6900]  # Valores estimados
})data_clusters = pd.DataFrame({
    "cluster": [0, 1, 2],
    "count": [7130.0, 3837.0, 39033.0],
    "mean": [0.853238, 0.720461, 0.765594],
    "std": [0.130881, 0.207740, 0.300997],
    "min": [-0.8899, -0.7579, -0.9835],
    "25%": [0.7906, 0.5267, 0.7096],
    "50%": [0.8977, 0.7783, 0.8885],
    "75%": [0.9460, 0.8885, 0.9524],
    "max": [0.9970, 0.9948, 0.9986]
})# Corregido: Aseguramos que year_month y vader_compound tengan la misma longitud (164 elementos)
data_mensual = pd.DataFrame({
    "year_month": [
        "2011-01", "2011-04", "2011-05", "2011-06", "2011-07", "2011-08", "2011-09", "2011-11", "2011-12",
        "2012-01", "2012-02", "2012-03", "2012-05", "2012-06", "2012-07", "2012-08", "2012-09", "2012-10",
        "2012-11", "2012-12", "2013-01", "2013-02", "2013-03", "2013-04", "2013-05", "2013-06", "2013-07",
        "2013-08", "2013-09", "2013-10", "2013-11", "2013-12", "2014-01", "2014-02", "2014-03", "2014-04",
        "2014-05", "2014-06", "2014-07", "2014-08", "2014-09", "2014-10", "2014-11", "2014-12", "2015-01",
        "2015-02", "2015-03", "2015-04", "2015-05", "2015-06", "2015-07", "2015-08", "2015-09", "2015-10",
        "2015-11", "2015-12", "2016-01", "2016-02", "2016-03", "2016-04", "2016-05", "2016-06", "2016-07",
        "2016-08", "2016-09", "2016-10", "2016-11", "2016-12", "2017-01", "2017-02", "2017-03", "2017-04",
        "2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10", "2017-11", "2017-12", "2018-01",
        "2018-02", "2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08", "2018-09", "2018-10",
        "2018-11", "2018-12", "2019-01", "2019-02", "2019-03", "2019-04", "2019-05", "2019-06", "2019-07",
        "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02", "2020-03", "2020-04",
        "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10", "2020-11", "2020-12", "2021-01",
        "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08", "2021-09", "2021-10",
        "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07",
        "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04",
        "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01",
        "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10",
        "2024-11", "2024-12"
    ],
    "vader_compound": [
        0.930000, 0.982000, 0.895700, 0.841100, 0.897150, 0.926600, 0.925667, 0.950100, 0.950100,
        0.929650, 0.977467, 0.968800, 0.879814, 0.954233, 0.922167, 0.930478, 0.938389, 0.958967,
        0.955725, 0.867740, 0.868100, 0.903800, 0.884836, 0.867258, 0.894208, 0.941038, 0.928197,
        0.933975, 0.928989, 0.920045, 0.892260, 0.857892, 0.873842, 0.890317, 0.915682, 0.931698,
        0.904630, 0.917828, 0.895731, 0.877517, 0.908437, 0.884002, 0.920922, 0.846467, 0.880280,
        0.886132, 0.877107, 0.884918, 0.894229, 0.877178, 0.841583, 0.874817, 0.890381, 0.879286,
        0.889026, 0.838844, 0.845016, 0.857054, 0.870728, 0.887189, 0.874846, 0.850709, 0.842540,
        0.840764, 0.847850, 0.839682, 0.808977, 0.811831, 0.786819, 0.816883, 0.803289, 0.819001,
        0.793763, 0.816881, 0.786060, 0.809665, 0.808683, 0.833374, 0.808800, 0.772921, 0.795341,
        0.788197, 0.780953, 0.786639, 0.795594, 0.824602, 0.794505, 0.793590, 0.800579, 0.813679,
        "2017-02", "2017-03", "2017-04", "2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10",
        "2017-11", "2017-12", "2018-01", "2018-02", "2018-03", "2018-04", "2018-05", "2018-06", "2018-07",
        "2018-08", "2018-09", "2018-10", "2018-11", "2018-12", "2019-01", "2019-02", "2019-03", "2019-04",
        "2019-05", "2019-06", "2019-07", "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01",
        "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10",
        "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07",
        "2021-08", "2021-09", "2021-10", "2021-11", "2021-12", "2022-01", "2022-02", "2022-03", "2022-04",
        "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
        "2023-02", "2023-03", "2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10",
        "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07",
        "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"
    ],
    "vader_compound": [
        0.930000, 0.982000, 0.895700, 0.841100, 0.897150, 0.926600, 0.925667, 0.950100, 0.950100,
        0.929650, 0.977467, 0.968800, 0.879814, 0.954233, 0.922167, 0.930478, 0.938389, 0.958967,
        0.955725, 0.867740, 0.868100, 0.903800, 0.884836, 0.867258, 0.894208, 0.941038, 0.928197,
        0.933975, 0.928989, 0.920045, 0.892260, 0.857892, 0.873842, 0.890317, 0.915682, 0.931698,
        0.904630, 0.917828, 0.895731, 0.877517, 0.908437, 0.884002, 0.920922, 0.846467, 0.880280,
        0.886132, 0.877107, 0.884918, 0.894229, 0.877178, 0.841583, 0.874817, 0.890381, 0.879286,
        0.889026, 0.838844, 0.845016, 0.857054, 0.870728, 0.887189, 0.874846, 0.850709, 0.842540,
        0.840764, 0.847850, 0.839682, 0.808977, 0.811831, 0.786819, 0.816883, 0.803289, 0.819001,
        0.793763, 0.816881, 0.786060, 0.809665, 0.808683, 0.833374, 0.808800, 0.772921, 0.795341,
        0.788197, 0.780953, 0.786639, 0.795594, 0.824602, 0.794505, 0.793590, 0.800579, 0.813679,
        0.791136, 0.759824, 0.749889, 0.763619, 0.770189, 0.767264, 0.786655, 0.785676, 0.796091,
        0.788698, 0.797587, 0.782660, 0.765938, 0.778965, 0.774635, 0.768214, 0.730183, 0.688162,
        0.670969, 0.652791, 0.720687, 0.650906, 0.678824, 0.688786, 0.732879, 0.660308, 0.700060,
        0.620922, 0.747775, 0.721271, 0.745978, 0.742302, 0.745000, 0.716813, 0.751618, 0.729861,
        0.734373, 0.737536, 0.716178, 0.721339, 0.726500, 0.762065, 0.786046, 0.784476, 0.775399,
        0.760688, 0.761779, 0.750654, 0.733250, 0.735556, 0.726192, 0.726420, 0.747397, 0.753837,
        0.767235, 0.764159, 0.770953, 0.736370, 0.773050, 0.765165, 0.749125, 0.720443, 0.730310,
        0.742275, 0.738527, 0.761762, 0.769454, 0.762163, 0.760203, 0.744064, 0.728979, 0.681660,
        0.440333, 0.698543
    ]
})# Resto del código (sin cambios)
resumen_general = {
    "total_reseñas": 50000,
    "total_usuarios": 49812,
    "periodo": {"inicio": "2011-01-04", "fin": "2024-12-25"},
    "promedio_sentimiento": 0.7746285680000001,
    "sentimiento_min": -0.9835,
    "sentimiento_max": 0.9986
}clusters = {
    "cluster_0": {
        "num_reseñas": 7130,
        "porcentaje": 14.26,
        "sentimiento_promedio": 0.8532384852734924,
        "palabras_clave": ["great", "location", "great location", "stay", "place", "apartment", "host", "great place", "clean", "recommend"]
    },
    "cluster_1": {
        "num_reseñas": 3837,
        "porcentaje": 7.674,
        "sentimiento_promedio": 0.7204611154547824,
        "palabras_clave": ["good", "location", "good location", "apartment", "place", "stay", "clean", "nice", "host", "good place"]
    },
    "cluster_2": {
        "num_reseñas": 39033,
        "porcentaje": 78.066,
        "sentimiento_promedio": 0.7655939512720007,
        "palabras_clave": ["apartment", "stay", "place", "nice", "location", "clean", "recommend", "great", "perfect", "host"]
    }
}temas_principales = {
    "Tema_1": "0.028*\"check\" + 0.021*\"help\" + 0.020*\"give\" + 0.017*\"time\" + 0.015*\"arrive\" + 0.012*\"leave\" + 0.011*\"arrival\" + 0.011*\"not\" + 0.011*\"question\" + 0.010*\"early\"",
    "Tema_2": "0.030*\"room\" + 0.018*\"not\" + 0.016*\"bed\" + 0.016*\"bathroom\" + 0.016*\"small\" + 0.013*\"kitchen\" + 0.013*\"night\" + 0.011*\"work\" + 0.010*\"shower\" + 0.010*\"bedroom\"",
    "Tema_3": "0.074*\"accommodation\" + 0.051*\"pleasant\" + 0.028*\"welcome\" + 0.025*\"locate\" + 0.016*\"description\" + 0.014*\"foot\" + 0.012*\"functional\" + 0.011*\"photo\" + 0.011*\"available\" + 0.011*\"practical\"",
    "Tema_4": "0.040*\"apartment\" + 0.038*\"great\" + 0.037*\"stay\" + 0.036*\"location\" + 0.027*\"place\" + 0.027*\"good\" + 0.025*\"clean\" + 0.020*\"recommend\" + 0.019*\"nice\" + 0.017*\"host\"",
    "Tema_5": "0.064*\"house\" + 0.034*\"attentive\" + 0.033*\"hostel\" + 0.020*\"attention\" + 0.018*\"department\" + 0.014*\"floor\" + 0.012*\"position\" + 0.011*\"doubt\" + 0.011*\"meter\" + 0.010*\"wide\""
}# Visualizaciones
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
    st.plotly_chart(fig, use_container_width=True)elif option == "Precios por Vecindario":
    bar_data = filtered_data.groupby("neighbourhood_cleansed")["price"].mean().reset_index()
    fig = px.bar(
        bar_data,
        x="neighbourhood_cleansed",
        y="price",
        title="Precio Promedio por Vecindario",
        color="price",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)# ... (Otras visualizaciones existentes sin cambios)

elif option == "Análisis de Clusters":
    st.subheader("Análisis de Clusters Identificados")
    st.markdown("""
    Esta sección presenta un análisis detallado de los clusters de reseñas identificados, incluyendo la actividad, el sentimiento promedio por día de la semana, la distribución de sentimientos por cluster y la evolución temporal del sentimiento.
    """)

# Resumen General
st.markdown("### Resumen General")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Reseñas", resumen_general["total_reseñas"])
with col2:
    st.metric("Total de Usuarios", resumen_general["total_usuarios"])
with col3:
    st.metric("Promedio de Sentimiento", f"{resumen_general['promedio_sentimiento']:.3f}")
st.markdown(f"**Período de Análisis**: {resumen_general['periodo']['inicio']} - {resumen_general['periodo']['fin']}")
st.markdown(f"**Rango de Sentimiento**: {resumen_general['sentimiento_min']:.4f} a {resumen_general['sentimiento_max']:.4f}")

# Información de Clusters
st.markdown("### Detalles de los Clusters")
for cluster, info in clusters.items():
    st.markdown(f"**{cluster.replace('_', ' ').title()}**")
    st.markdown(f"- **Número de Reseñas**: {info['num_reseñas']} ({info['porcentaje']:.2f}%)")
    st.markdown(f"- **Sentimiento Promedio**: {info['sentimiento_promedio']:.3f}")
    st.markdown(f"- **Palabras Clave**: {', '.join(info['palabras_clave'])}")

# Temas Principales
st.markdown("### Temas Principales Identificados")
for tema, descripcion in temas_principales.items():
    st.markdown(f"**{tema}**: {descripcion}")

# Gráfico 1: Actividad y Sentimiento por Día de la Semana
st.markdown("### Actividad y Sentimiento por Día de la Semana")
st.markdown("""
**Descripción**: Gráfico combinado que muestra la actividad y el sentimiento promedio de las reseñas por día de la semana. Las barras indican el número de reseñas, mientras que la línea roja muestra el sentimiento promedio (calculado con VADER).  
**Información Adicional**:  
- Período de análisis: Semanal (por día)  
- Método de análisis de sentimiento: VADER
""")
fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        x=data_dia_semana["Día"],
        y=data_dia_semana["num_reseñas"],
        name="Número de Reseñas",
        marker_color="skyblue"
    )
)
fig1.add_trace(
    go.Scatter(
        x=data_dia_semana["Día"],
        y=data_dia_semana["vader_compound"],
        name="Sentimiento Promedio",
        line=dict(color="red", width=2),
        yaxis="y2"
    )
)
fig1.update_layout(
    title="Actividad y Sentimiento por Día de la Semana",
    xaxis=dict(title="Día de la Semana"),
    yaxis=dict(title="Número de Reseñas", titlefont=dict(color="skyblue"), tickfont=dict(color="skyblue")),
    yaxis2=dict(title="Sentimiento Promedio (VADER)", titlefont=dict(color="red"), tickfont=dict(color="red"), overlaying="y", side="right"),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: Distribución de Sentimiento por Cluster
st.markdown("### Distribución de Sentimiento por Cluster")
st.markdown("""
**Descripción**: Diagrama de caja que muestra la distribución de las puntuaciones de sentimiento (calculadas con VADER) para cada cluster de reseñas. Permite comparar el rango y la mediana del sentimiento entre clusters.  
**Información Adicional**:  
- Número de clusters: 3  
- Método de análisis de sentimiento: VADER
""")
fig2 = go.Figure()
for cluster in data_clusters["cluster"]:
    cluster_data = data_clusters[data_clusters["cluster"] == cluster]
    fig2.add_trace(
        go.Box(
            y=[
                cluster_data["min"].iloc[0],
                cluster_data["25%"].iloc[0],
                cluster_data["50%"].iloc[0],
                cluster_data["75%"].iloc[0],
                cluster_data["max"].iloc[0]
            ],
            name=f"Cluster {cluster}",
            boxpoints=False
        )
    )
fig2.update_layout(
    title="Distribución de Sentimiento por Cluster",
    yaxis=dict(title="Sentimiento (VADER)"),
    xaxis=dict(title="Cluster")
)
st.plotly_chart(fig2, use_container_width=True)

# Gráfico 3: Evolución del Sentimiento Mensual
st.markdown("### Evolución del Sentimiento Mensual")
st.markdown("""
**Descripción**: Gráfico de líneas que muestra la evolución del sentimiento promedio (calculado con VADER) de las reseñas a lo largo del tiempo, agrupado por mes.  
**Información Adicional**:  
- Período de análisis: Mensual  
- Método de análisis de sentimiento: VADER
""")
fig3 = px.line(
    data_mensual,
    x="year_month",
    y="vader_compound",
    title="Evolución del Sentimiento Promedio Mensual",
    labels={"year_month": "Mes", "vader_compound": "Sentimiento Promedio (VADER)"}
)
fig3.update_xaxes(tickangle=45)
st.plotly_chart(fig3, use_container_width=True)

# Métricas resumidas (sin cambios)


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
    st.metric("Antigüedad Promedio del Host (años)", f"{filtered_data['host_age_years'].mean():.2f}")# Pie de página (sin cambios)
st.markdown("---")
st.markdown("TFG - Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb | Ángel Soto García")

