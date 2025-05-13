# Barra lateral para navegación y filtros
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/2560px-Airbnb_Logo_B%C3%A9lo.svg.png", width=200)
    
    st.markdown("### Navegación")
    page = st.radio(
        "Selecciona una sección:",
        ["📊 Resumen General", 
         "💰 Análisis de Precios", 
         "📍 Análisis Geográfico", 
         "👥 Segmentación de Usuarios",
         "📝 Análisis de Reseñas"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### Filtros")
    st.markdown("Los filtros se aplicarán cuando se carguen los datos.")
    
    # Placeholder para filtros futuros
    st.markdown("#### Ciudad")
    ciudades_filter = st.multiselect(
        "Selecciona ciudades:",
        ["Madrid", "Barcelona", "Valencia", "Sevilla", "Málaga"],
        default=["Madrid", "Barcelona"]
    )
    
    st.markdown("#### Tipo de Alojamiento")
    tipo_filter = st.multiselect(
        "Selecciona tipos:",
        ["Apartamento entero", "Habitación privada", "Habitación compartida", "Casa entera"],
        default=["Apartamento entero", "Casa entera"]
    )
    
    st.markdown("#### Rango de Precio")
    price_range = st.slider(
        "Precio por noche (€):",
        0, 500, (50, 200)
    )
    
    st.markdown("#### Calificación")
    rating_filter = st.slider(
        "Calificación mínima:",
        1.0, 5.0, 4.0, 0.1
    )
    
    st.markdown("---")
    
    st.markdown("### Acerca del Autor")
    st.markdown("""
    **Ángel Soto García**  
    Grado en Ciencia de datos  
    Universidad Oberta de Catalunya  
    2025
    """)

# Contenido principal
st.markdown('<h1 class="main-header">Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #767676;">Una perspectiva desde la ciencia de datos</h3>', unsafe_allow_html=True)

# Sección de introducción
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Introducción</h2>', unsafe_allow_html=True)
st.markdown("""
Este dashboard interactivo presenta los resultados del análisis de datos de Airbnb realizado como parte del Trabajo de Fin de Grado en Ciencia de Datos. La investigación se centra en:

1. **Predicción de precios** de alojamientos utilizando modelos de aprendizaje automático.
2. **Análisis exploratorio** para comprender la estructura del conjunto de datos.
3. **Análisis de reseñas** mediante procesamiento de lenguaje natural.

Utiliza los filtros en la barra lateral para personalizar la visualización y navega por las diferentes secciones para explorar los resultados del análisis.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Dependiendo de la página seleccionada, mostrar diferentes secciones
if page == "📊 Resumen General":
    st.markdown('<h2 class="sub-header">Resumen General</h2>', unsafe_allow_html=True)
    
    # Métricas principales - Placeholders
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Total Alojamientos", value="0")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Precio Medio", value="0 €")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Calificación Media", value="0.0")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="% Superhosts", value="0%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Secciones de análisis rápido
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">Distribución de Precios</h3>', unsafe_allow_html=True)
        st.markdown("*Se mostrará un histograma con la distribución de precios de los alojamientos.*")
        # Placeholder para gráfico
        st.image("https://via.placeholder.com/600x400?text=Histograma+de+Precios", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">Tipos de Alojamiento</h3>', unsafe_allow_html=True)
        st.markdown("*Se mostrará un gráfico de torta con la distribución de tipos de alojamiento.*")
        # Placeholder para gráfico
        st.image("https://via.placeholder.com/600x400?text=Distribución+por+Tipo", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Características principales
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Características más Valoradas</h3>', unsafe_allow_html=True)
    st.markdown("*Se mostrará un gráfico de barras con las características más valoradas según las reseñas.*")
    # Placeholder para gráfico
    st.image("https://via.placeholder.com/1200x400?text=Características+más+Valoradas", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # KPIs por ciudad
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Comparativa por Ciudades</h3>', unsafe_allow_html=True)
    st.markdown("*Se mostrará una tabla comparativa con los principales KPIs por ciudad.*")
    # Placeholder para tabla
    st.image("https://via.placeholder.com/1200x300?text=Tabla+Comparativa+por+Ciudades", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "💰 Análisis de Precios":
    st.markdown('<h2 class="sub-header">Análisis de Precios</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Esta sección muestra el análisis detallado de los precios y los factores que los influencian.</div>', unsafe_allow_html=True)
    
    # Contenido placeholder para la sección de precios
    st.markdown("*El contenido de la sección de análisis de precios se cargará cuando se implementen los datos.*")
    
    # Subsección de modelado predictivo
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Modelado Predictivo de Precios</h3>', unsafe_allow_html=True)
    st.markdown("*Aquí se mostrarán los resultados del modelo predictivo de precios, incluyendo métricas de rendimiento e importancia de características.*")
    # Placeholder para gráfico
    st.image("https://via.placeholder.com/1200x400?text=Importancia+de+Características", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "📍 Análisis Geográfico":
    st.markdown('<h2 class="sub-header">Análisis Geográfico</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Esta sección visualiza la distribución geográfica de los alojamientos y su relación con los precios y calificaciones.</div>', unsafe_allow_html=True)
    
    # Contenido placeholder para la sección geográfica
    st.markdown("*El contenido de la sección de análisis geográfico se cargará cuando se implementen los datos.*")
    
    # Placeholder para mapa
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Mapa de Calor de Precios</h3>', unsafe_allow_html=True)
    st.markdown("*Aquí se mostrará un mapa de calor con la distribución de precios por ubicación.*")
    # Placeholder para mapa
    st.image("https://via.placeholder.com/1200x600?text=Mapa+de+Calor", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "👥 Segmentación de Usuarios":
    st.markdown('<h2 class="sub-header">Segmentación de Usuarios</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Esta sección presenta los resultados del análisis de segmentación de usuarios basado en sus preferencias y comportamientos.</div>', unsafe_allow_html=True)
    
    # Contenido placeholder para la sección de segmentación
    st.markdown("*El contenido de la sección de segmentación de usuarios se cargará cuando se implementen los datos.*")
    
    # Placeholder para clusters
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Clusters de Usuarios</h3>', unsafe_allow_html=True)
    st.markdown("*Aquí se mostrará una visualización de los clusters de usuarios identificados.*")
    # Placeholder para gráfico
    st.image("https://via.placeholder.com/1200x600?text=Visualización+de+Clusters", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:  # Análisis de Reseñas
    st.markdown('<h2 class="sub-header">Análisis de Reseñas</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Esta sección presenta el análisis de sentimientos y temas extraídos de las reseñas de los usuarios.</div>', unsafe_allow_html=True)
    
    # Contenido placeholder para la sección de reseñas
    st.markdown("*El contenido de la sección de análisis de reseñas se cargará cuando se implementen los datos.*")
    
    # Placeholder para análisis de sentimientos
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Análisis de Sentimientos</h3>', unsafe_allow_html=True)
    st.markdown("*Aquí se mostrará un gráfico con la distribución de sentimientos en las reseñas.*")
    # Placeholder para gráfico
    st.image("https://via.placeholder.com/1200x400?text=Análisis+de+Sentimientos", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Placeholder para nube de palabras
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Términos más Frecuentes</h3>', unsafe_allow_html=True)
    st.markdown("*Aquí se mostrará una nube de palabras con los términos más frecuentes en las reseñas.*")
    # Placeholder para nube de palabras
    st.image("https://via.placeholder.com/1200x600?text=Nube+de+Palabras", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Pie de página
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
**Análisis Predictivo de Precios y Segmentación de Usuarios en Airbnb**  
Trabajo Final de Grado | Universitat Oberta de Catalunya | 2025
""")
st.markdown('</div>', unsafe_allow_html=True)import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Análisis Airbnb - TFG",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores Airbnb
COLOR_AIRBNB_PRIMARY = "#FF5A5F"       # Rojo Airbnb
COLOR_AIRBNB_SECONDARY = "#00A699"     # Verde turquesa
COLOR_AIRBNB_TERTIARY = "#FC642D"      # Naranja
COLOR_AIRBNB_QUATERNARY = "#484848"    # Gris oscuro
COLOR_AIRBNB_LIGHT = "#767676"         # Gris claro

# Aplicar estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5A5F;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #484848;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #767676;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        font-weight: bold;
    }
    .highlight-text {
        background-color: #F7F7F7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF5A5F;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EDFBFF;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #00A699;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #DDDDDD;
        color: #767676;
    }
    /* Personalizaciones adicionales */
    div.stButton > button {
        background-color: #FF5A5F;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    div.stButton > button:hover {
        background-color: #FC642D;
    }
    .stSelectbox label, .stSlider label {
        color: #484848;
        font-weight: 500;
    }
    .stProgress > div > div > div > div {
        background-color: #00A699;
    }
</style>
""", unsafe_allow_html=True)
