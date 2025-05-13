# Importaciones de bibliotecas estándar
from pathlib import Path
from datetime import datetime
import json
import os
import numpy as np

# Importaciones de bibliotecas de terceros
import pandas as pd
from langdetect import detect, LangDetectException
import re
import spacy
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Importaciones para análisis de sentimiento
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Asegurar disponibilidad del lexicón VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Descargando 'vader_lexicon' de NLTK...")
    nltk.download('vader_lexicon')

try:
    from transformers import pipeline
    transformer_available = True
except ImportError:
    print("Advertencia: 'transformers' no está disponible. Se omitirá el análisis de sentimiento con Transformers.")
    transformer_available = False

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Función  para generar archivos de texto descriptivos para cada PNG
def generate_text_for_png(png_filename, description, additional_info=None, data=None):
    """
    Genera un archivo de texto (.txt) correspondiente a un archivo PNG con una descripción de su contenido
    y los datos utilizados para generar el gráfico.
    
    Args:
        png_filename (str): Nombre del archivo PNG (sin la extensión).
        description (str): Descripción del contenido del gráfico.
        additional_info (dict, optional): Información adicional para incluir en el archivo de texto.
        data (dict, list, pd.DataFrame, np.ndarray, optional): Datos específicos del gráfico.
    """
    txt_filename = f"{png_filename}.txt"
    content = f"Archivo gráfico: {png_filename}.png\n\n"
    content += f"Descripción:\n{description}\n\n"
    
    if additional_info:
        content += "Información adicional:\n"
        for key, value in additional_info.items():
            content += f"  {key}: {value}\n"
    
    if data is not None:
        content += "\nDatos del gráfico:\n"
        if isinstance(data, pd.DataFrame):
            content += data.to_string(index=True) + "\n"
        elif isinstance(data, np.ndarray):
            content += f"Array NumPy con forma {data.shape}:\n"
            content += np.array2string(data, precision=4, suppress_small=True, max_line_width=100) + "\n"
        elif isinstance(data, dict):
            for key, value in data.items():
                content += f"  {key}:\n"
                if isinstance(value, list):
                    content += f"    {', '.join(map(str, value[:10]))}"
                    if len(value) > 10:
                        content += f"... (mostrando solo los primeros 10 de {len(value)} elementos)\n"
                    else:
                        content += "\n"
                elif isinstance(value, (pd.Series, np.ndarray)):
                    content += f"    {str(value[:10])}\n"
                    if len(value) > 10:
                        content += f"    ... (mostrando solo los primeros 10 de {len(value)} elementos)\n"
                else:
                    content += f"    {value}\n"
        elif isinstance(data, list):
            content += f"  {', '.join(map(str, data[:10]))}"
            if len(data) > 10:
                content += f"... (mostrando solo los primeros 10 de {len(data)} elementos)\n"
            else:
                content += "\n"
        else:
            content += f"  {str(data)}\n"
    
    content += f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Archivo de texto generado: {txt_filename}")

# Función para cargar datos
def load_data(file_name):
    """
    Carga datos desde un archivo CSV utilizando un enfoque robusto de resolución de rutas.
    Intenta encontrar el archivo en varias de las ubicaciones comunes.
    """
    possible_paths = [
        file_name,
        Path.cwd() / file_name,
        Path.cwd() / 'datasets' / file_name,
    ]
    try:
        script_path = Path(__file__).resolve().parent
        possible_paths.extend([
            script_path / file_name,
            script_path / 'datasets' / file_name,
            script_path.parent / 'datasets' / file_name,
            script_path.parent.parent / 'datasets' / file_name,
        ])
    except NameError:
        pass

    file_path = next((p for p in possible_paths if Path(p).exists()), None)
    if file_path is None:
        raise FileNotFoundError(f"No se pudo encontrar {file_name} en ninguna de las ubicaciones esperadas")

    col_names = ['listing_id', 'review_id', 'date', 'reviewer_id', 'reviewer_name', 'comments']
    df = pd.read_csv(file_path, sep='\t', names=col_names, header=None)
    print(f"Datos cargados correctamente desde {file_path}. Shape: {df.shape}")
    
    # Convertir fecha a formato datetime
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"Error al convertir fechas: {e}")
    
    return df

# Función para detectar el idioma
def detect_language(text):
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

# Función para preprocesar el texto
def preprocess_text(text, nlp, keep_entities=True):
    if not isinstance(text, str) or not text.strip():
        return ''
    # Eliminar HTML y URLs
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    # Mantener solo letras y espacios
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    # Tokenización y entidades
    doc = nlp(text)
    tokens = []
    for token in doc:
        if keep_entities and token.ent_type_ in ['PERSON', 'GPE', 'ORG']:
            tag = f'[{token.ent_type_}]'
            if not tokens or tokens[-1] != tag:
                tokens.append(tag)
        elif token.is_alpha and not token.is_stop:
            tokens.append(token.lemma_)
    return ' '.join(tokens)

# Función para análisis de sentimiento
def analyze_sentiment(text, vader_analyzer, transformer_pipeline=None):
    """Analiza el sentimiento de un texto usando VADER y opcionalmente transformers"""
    if not isinstance(text, str) or not text.strip():
        return {'vader': 0, 'transformer_label': 'NEUTRAL', 'transformer_score': 0.5}
    
    result = {}
    # Análisis VADER
    vader_scores = vader_analyzer.polarity_scores(text)
    result['vader'] = vader_scores['compound']
    
    # Análisis con transformers si está disponible
    if transformer_pipeline:
        try:
            tr = transformer_pipeline(text[:512])[0]
            result['transformer_label'] = tr['label']
            result['transformer_score'] = tr['score']
        except Exception:
            result['transformer_label'] = 'ERROR'
            result['transformer_score'] = 0.5
    else:
        result['transformer_label'] = 'N/A'
        result['transformer_score'] = None
    
    return result

# Función para generar características de usuario
def generate_user_features(df):
    """Genera características a nivel de usuario basadas en sus reseñas"""
    user_features = df.groupby('reviewer_id').agg({
        'review_id': 'count',
        'date': ['min', 'max'],
        'vader_compound': ['mean', 'std', 'min', 'max'],
        'cluster': lambda x: Counter(x).most_common(1)[0][0],
        # Modificación: Asegurar que solo se unan strings válidos en comments
        'comments': lambda x: ' '.join([str(comment) for comment in x if isinstance(comment, str) and comment.strip()])
    })
    
    # Aplanar columnas multi-índice
    user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
    
    # Calcular tiempo activo en días
    user_features['days_active'] = (user_features['date_max'] - user_features['date_min']).dt.days
    
    # Longitud promedio de reseñas
    user_features['avg_review_length'] = df.groupby('reviewer_id')['comments'].apply(
        lambda x: np.mean([len(str(comment)) for comment in x if isinstance(comment, str)])
    )
    
    return user_features

# Función para extraer temas mediante LDA
def extract_topics(texts, num_topics=5, words_per_topic=10):
    """Extrae temas utilizando LDA (Latent Dirichlet Allocation)"""
    # Tokenizar
    tokenized_texts = [text.split() for text in texts if isinstance(text, str) and text.strip()]
    
    # Crear diccionario y corpus
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # Entrenar modelo LDA
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        alpha='auto',
        random_state=42
    )
    
    # Extraer temas
    topics = {}
    for idx, topic in lda_model.print_topics(num_words=words_per_topic):
        topics[f"Tema_{idx+1}"] = topic
    
    return topics, lda_model, dictionary

# Función para visualizar resultados
def visualize_results(df, reduced_data, labels, k, vectorizer, user_features=None):
    """Genera visualizaciones para los resultados del análisis, archivos de texto y datos asociados"""
    # 1. Visualización de clusters
    plt.figure(figsize=(10, 8))
    cluster_data = {}
    for label in range(k):
        pts = reduced_data[labels == label]
        plt.scatter(pts[:,0], pts[:,1], label=f'Cluster {label}', alpha=0.4)
        cluster_data[f"Cluster_{label}_points"] = pts.tolist()
    
    centroids = KMeans(n_clusters=k, n_init=10, random_state=42).fit(reduced_data).cluster_centers_
    plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, c='red', label='Centroides')
    cluster_data["Centroides"] = centroids.tolist()
    plt.title('Clusters de Reseñas en Espacio PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.savefig('clusters_pca.png')
    plt.close()
    
    # Generar archivo de texto para clusters_pca.png
    generate_text_for_png(
        'clusters_pca',
        "Gráfico de dispersión que muestra los clusters de reseñas en un espacio bidimensional reducido mediante PCA. Cada punto representa una reseña, coloreada según su cluster asignado por el algoritmo K-Means. Las 'X' rojas indican los centroides de los clusters.",
        {
            "Número de clusters": k,
            "Método de reducción dimensional": "PCA",
            "Algoritmo de clustering": "K-Means"
        },
        data=cluster_data
    )
    
    # 2. Análisis de sentimiento por cluster
    sentiment_data = df.groupby('cluster')['vader_compound'].describe()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='vader_compound', data=df)
    plt.title('Distribución de Sentimiento por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Puntuación de Sentimiento (VADER)')
    plt.savefig('sentiment_by_cluster.png')
    plt.close()
    
    # Generar archivo de texto para sentiment_by_cluster.png
    generate_text_for_png(
        'sentiment_by_cluster',
        "Diagrama de caja que muestra la distribución de las puntuaciones de sentimiento (calculadas con VADER) para cada cluster de reseñas. Permite comparar el rango y la mediana del sentimiento entre clusters.",
        {
            "Número de clusters": k,
            "Método de análisis de sentimiento": "VADER"
        },
        data=sentiment_data
    )
    
    # 3. Nubes de palabras por cluster
    terms = vectorizer.get_feature_names_out()
    for cluster_id in range(k):
        cluster_texts = ' '.join(df[df['cluster'] == cluster_id]['preprocessed_comments'])
        if not cluster_texts.strip():
            continue
            
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(cluster_texts)
        
        # Obtener frecuencias de palabras
        word_freq = wordcloud.words_
        word_freq_sorted = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Palabras Clave del Cluster {cluster_id}')
        plt.savefig(f'wordcloud_cluster_{cluster_id}.png')
        plt.close()
        
        # Generar archivo de texto para wordcloud_cluster_{cluster_id}.png
        generate_text_for_png(
            f'wordcloud_cluster_{cluster_id}',
            f"Nube de palabras que muestra los términos más frecuentes y relevantes en las reseñas del cluster {cluster_id}, basada en los comentarios preprocesados.",
            {
                "Cluster ID": cluster_id,
                "Número máximo de palabras": 100,
                "Preprocesamiento": "Eliminación de stop words, lematización, y filtrado de términos no alfabéticos"
            },
            data={"Palabras más frecuentes (top 10)": word_freq_sorted}
        )
    
    # 4. Análisis temporal si hay fechas disponibles
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['year_month'] = df['date'].dt.to_period('M')
        sentiment_over_time = df.groupby('year_month')['vader_compound'].mean().reset_index()
        sentiment_over_time['year_month'] = sentiment_over_time['year_month'].astype(str)
        
        plt.figure(figsize=(12, 6))
        sentiment_over_time.plot(kind='line', x='year_month', y='vader_compound', marker='o')
        plt.title('Evolución del Sentimiento a lo Largo del Tiempo')
        plt.xlabel('Fecha')
        plt.ylabel('Sentimiento Promedio')
        plt.grid(True, alpha=0.3)
        plt.savefig('sentiment_over_time.png')
        plt.close()
        
        # Generar archivo de texto para sentiment_over_time.png
        generate_text_for_png(
            'sentiment_over_time',
            "Gráfico de líneas que muestra la evolución del sentimiento promedio (calculado con VADER) de las reseñas a lo largo del tiempo, agrupado por mes.",
            {
                "Período de análisis": "Mensual",
                "Método de análisis de sentimiento": "VADER"
            },
            data=sentiment_over_time
        )
    
    # 5. Segmentación de usuarios si hay datos disponibles
    if user_features is not None and len(user_features) > 5:
        # Seleccionar características para segmentación
        user_seg_features = user_features[[
            'review_id_count', 'vader_compound_mean', 
            'days_active', 'avg_review_length'
        ]].copy()
        
        # Normalizar
        scaler = StandardScaler()
        user_seg_scaled = scaler.fit_transform(user_seg_features.fillna(0))
        
        # PCA para visualización
        user_pca = PCA(n_components=2, random_state=42).fit_transform(user_seg_scaled)
        
        # DBSCAN para segmentación
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        user_segments = dbscan.fit_predict(user_seg_scaled)
        
        # Preparar datos para el archivo de texto
        user_segment_data = {
            "Puntos_PCA": user_pca.tolist(),
            "Segmentos": user_segments.tolist(),
            "Outliers": user_pca[user_segments == -1].tolist()
        }
        
        # Visualizar
        plt.figure(figsize=(10, 8))
        mask_valid = user_segments >= 0
        for segment in np.unique(user_segments[mask_valid]):
            pts = user_pca[mask_valid & (user_segments == segment)]
            plt.scatter(pts[:,0], pts[:,1], label=f'Segmento {segment}', alpha=0.7)
        
        outliers = user_pca[user_segments == -1]
        if len(outliers) > 0:
            plt.scatter(outliers[:,0], outliers[:,1], label='Outliers', 
                        color='black', marker='x', alpha=0.5)
        
        plt.title('Segmentación de Usuarios')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend()
        plt.savefig('user_segments.png')
        plt.close()
        
        # Generar archivo de texto para user_segments.png
        generate_text_for_png(
            'user_segments',
            "Gráfico de dispersión que muestra la segmentación de usuarios en un espacio bidimensional reducido mediante PCA. Los puntos representan usuarios, coloreados según su segmento asignado por DBSCAN. Los 'x' negros indican outliers.",
            {
                "Algoritmo de segmentación": "DBSCAN",
                "Método de reducción dimensional": "PCA",
                "Características usadas": "Número de reseñas, promedio de sentimiento, días activos, longitud promedio de reseñas"
            },
            data=user_segment_data
        )

# Función para generar informe
def generate_report(df, user_features, topics, k):
    """Genera un informe resumido de los resultados del análisis"""
    report = {
        "resumen_general": {
            "total_reseñas": len(df),
            "total_usuarios": len(df['reviewer_id'].unique()),
            "periodo": {
                "inicio": df['date'].min().strftime('%Y-%m-%d'),
                "fin": df['date'].max().strftime('%Y-%m-%d')
            },
            "promedio_sentimiento": float(df['vader_compound'].mean()),
            "sentimiento_min": float(df['vader_compound'].min()),
            "sentimiento_max": float(df['vader_compound'].max())
        },
        "clusters": {},
        "temas_principales": topics,
        "segmentos_usuario": {}
    }
    
    # Información por cluster
    for cluster_id in range(k):
        cluster_data = df[df['cluster'] == cluster_id]
        if len(cluster_data) == 0:
            continue
            
        report["clusters"][f"cluster_{cluster_id}"] = {
            "num_reseñas": len(cluster_data),
            "porcentaje": float(len(cluster_data) / len(df) * 100),
            "sentimiento_promedio": float(cluster_data['vader_compound'].mean()),
            "palabras_clave": list(extract_top_tfidf_terms(cluster_data['preprocessed_comments']))[:10]
        }
    
    # Guardar informe como JSON
    with open('informe_analisis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report

# Función auxiliar para extraer términos TF-IDF importantes
def extract_top_tfidf_terms(texts, n_top=20):
    """Extrae los términos más importantes según TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
    try:
        X = vectorizer.fit_transform(texts)
        importance = np.argsort(np.asarray(X.sum(axis=0)).ravel())[::-1]
        feature_names = np.array(vectorizer.get_feature_names_out())
        return feature_names[importance[:n_top]]
    except:
        return []

# Función principal de análisis
def analyze_reviews(file_name, max_reviews=25000, k_clusters=3, analyze_users=True):
    """Función principal que coordina el análisis completo"""
    print("\n=== INICIANDO ANÁLISIS DE RESEÑAS ===\n")
    
    # Cargar y preprocesar datos
    df_full = load_data(file_name)
    if df_full.empty:
        raise SystemExit("DataFrame vacío, verifique el archivo de datos.")
    
    # Limitar a número máximo de reseñas
    if len(df_full) > max_reviews:
        df_full = df_full.head(max_reviews)
        print(f"Limitando análisis a {max_reviews} reseñas")
    
    total = len(df_full)
    print(f"\nProcesando {total} reseñas...\n")
    
    # Cargar recursos NLP
    nlp = spacy.load('en_core_web_sm')
    translator = GoogleTranslator(source='auto', target='en')
    vader_analyzer = SentimentIntensityAnalyzer()
    transformer_pipeline = None
    if transformer_available:
        try:
            transformer_pipeline = pipeline('sentiment-analysis')
        except Exception as e:
            print(f"Error al cargar pipeline de transformers: {e}")
    
    # Procesamiento por lotes
    print("1. Detección de idioma y preprocesamiento...")
    languages, preprocessed = [], []
    for i, txt in enumerate(df_full['comments'], 1):
        if i % 100 == 0 or i == total:
            print(f"  Procesando reseña {i}/{total}")
        
        # Detectar idioma
        lang = detect_language(txt)
        languages.append(lang)
        
        # Traducir si no está en inglés
        if lang != 'en' and isinstance(txt, str) and txt.strip():
            try:
                txt = translator.translate(txt)
            except Exception as e:
                print(f"Error traduciendo reseña {i}: {e}")
        
        # Preprocesar texto
        preprocessed.append(preprocess_text(txt, nlp))
    
    # Asignar al DataFrame
    df_full['language'] = languages
    df_full['preprocessed_comments'] = preprocessed
    
    # Análisis de sentimiento
    print("\n2. Realizando análisis de sentimiento...")
    sentiment_results = []
    for i, text in enumerate(df_full['preprocessed_comments'], 1):
        if i % 100 == 0 or i == total:
            print(f"  Analizando sentimiento reseña {i}/{total}")
        result = analyze_sentiment(text, vader_analyzer, transformer_pipeline)
        sentiment_results.append(result)
    
    # Asignar resultados de sentimiento al DataFrame
    df_full['vader_compound'] = [result['vader'] for result in sentiment_results]
    if transformer_available:
        df_full['sent_label'] = [result['transformer_label'] for result in sentiment_results]
        df_full['sent_score'] = [result['transformer_score'] for result in sentiment_results]
    
    # Vectorización TF-IDF
    print("\n3. Vectorización TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
    X_tfidf = vectorizer.fit_transform(df_full['preprocessed_comments']).toarray()
    
    # PCA para visualización
    print("4. Reducción dimensional con PCA...")
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(X_tfidf)
    
    # Clustering K-Means
    print(f"5. Clustering K-Means (K={k_clusters})...")
    kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_tfidf)  # Usar vectores TF-IDF completos para clustering
    df_full['cluster'] = labels
    
    # Extraer temas con LDA
    print("\n6. Extrayendo temas principales con LDA...")
    topics, lda_model, dictionary = extract_topics(
        df_full['preprocessed_comments'], 
        num_topics=5
    )
    print("Temas extraídos:")
    for topic_id, topic_desc in topics.items():
        print(f"  {topic_id}: {topic_desc[:100]}...")
    
    # Análisis de usuarios
    user_features = None
    if analyze_users and len(df_full['reviewer_id'].unique()) > 5:
        print("\n7. Generando perfiles de usuarios...")
        user_features = generate_user_features(df_full)
        print(f"  Se generaron perfiles para {len(user_features)} usuarios")
    
    # Visualizaciones
    print("\n8. Generando visualizaciones...")
    visualize_results(df_full, reduced, labels, k_clusters, vectorizer, user_features)
    
    # Generar informe
    print("\n9. Generando informe de resultados...")
    report = generate_report(df_full, user_features, topics, k_clusters)
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Archivos generados en el directorio actual:")
    print("  - clusters_pca.png (con clusters_pca.txt)")
    print("  - sentiment_by_cluster.png (con sentiment_by_cluster.txt)")
    for i in range(k_clusters):
        print(f"  - wordcloud_cluster_{i}.png (con wordcloud_cluster_{i}.txt)")
    print("  - sentiment_over_time.png (con sentiment_over_time.txt)")
    if user_features is not None:
        print("  - user_segments.png (con user_segments.txt)")
    print("  - informe_analisis.json")
    
    return df_full, user_features, report

# Función para encontrar el K óptimo para clustering
def find_optimal_k(data, k_range=range(2, 10)):
    """Encuentra el número óptimo de clusters usando múltiples métricas"""
    scores = {
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data)
        
        scores['silhouette'].append(silhouette_score(data, labels))
        scores['davies_bouldin'].append(davies_bouldin_score(data, labels))
        scores['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
    
    # Preparar datos para el gráfico
    scores_df = pd.DataFrame({
        'K': list(k_range),
        'Silhouette': scores['silhouette'],
        'Davies_Bouldin': scores['davies_bouldin'],
        'Calinski_Harabasz': scores['calinski_harabasz']
    })
    
    # Visualizar resultados
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(k_range, scores['silhouette'], marker='o')
    plt.title('Silhouette Score vs K')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')
    
    plt.subplot(1, 3, 2)
    plt.plot(k_range, scores['davies_bouldin'], marker='o')
    plt.title('Davies-Bouldin Index vs K')
    plt.xlabel('K')
    plt.ylabel('Davies-Bouldin Index')
    
    plt.subplot(1, 3, 3)
    plt.plot(k_range, scores['calinski_harabasz'], marker='o')
    plt.title('Calinski-Harabasz Index vs K')
    plt.xlabel('K')
    plt.ylabel('Calinski-Harabasz Index')
    
    plt.tight_layout()
    plt.savefig('optimal_k.png')
    plt.close()
    
    # Generar archivo de texto para optimal_k.png
    generate_text_for_png(
        'optimal_k',
        "Conjunto de tres gráficos que muestran métricas de evaluación para determinar el número óptimo de clusters (K) en el algoritmo K-Means. Incluye Silhouette Score (mayor es mejor), Davies-Bouldin Index (menor es mejor), y Calinski-Harabasz Index (mayor es mejor).",
        {
            "Rango de K evaluado": f"{min(k_range)}-{max(k_range)}",
            "Algoritmo de clustering": "K-Means",
            "Métricas usadas": "Silhouette, Davies-Bouldin, Calinski-Harabasz"
        },
        data=scores_df
    )
    
    # Determinar K óptimo
    k_sil = k_range[np.argmax(scores['silhouette'])]
    k_db = k_range[np.argmin(scores['davies_bouldin'])]
    k_ch = k_range[np.argmax(scores['calinski_harabasz'])]
    
    print(f"K óptimo según Silhouette: {k_sil}")
    print(f"K óptimo según Davies-Bouldin: {k_db}")
    print(f"K óptimo según Calinski-Harabasz: {k_ch}")
    
    # Retornar un consenso o el valor más frecuente
    k_values = [k_sil, k_db, k_ch]
    k_optimal = max(set(k_values), key=k_values.count)
    
    return k_optimal, scores

# Función para analizar tendencias temporales
def analyze_temporal_trends(df):
    """Analiza tendencias temporales en las reseñas"""
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        print("Error: Las fechas no están en formato datetime")
        return None
    
    print("\nAnalizando tendencias temporales...")
    
    # Agregar columnas de tiempo
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Evolución mensual
    monthly_stats = df.groupby(df['date'].dt.to_period('M')).agg({
        'review_id': 'count',
        'vader_compound': 'mean',
        'cluster': lambda x: x.value_counts().index[0]
    }).reset_index()
    monthly_stats['date'] = monthly_stats['date'].dt.to_timestamp()
    
    # Visualizar tendencias
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Volumen de reseñas
    ax[0].plot(monthly_stats['date'], monthly_stats['review_id'], marker='o')
    ax[0].set_title('Volumen mensual de reseñas')
    ax[0].set_ylabel('Número de reseñas')
    ax[0].grid(True, alpha=0.3)
    
    # Sentimiento
    ax[1].plot(monthly_stats['date'], monthly_stats['vader_compound'], marker='o', color='green')
    ax[1].set_title('Evolución del sentimiento mensual')
    ax[1].set_ylabel('Sentimiento promedio')
    ax[1].set_xlabel('Fecha')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tendencias_temporales.png')
    plt.close()
    
    # Generar archivo de texto para tendencias_temporales.png
    generate_text_for_png(
        'tendencias_temporales',
        "Gráfico compuesto por dos subgráficos que muestran tendencias temporales de las reseñas. El primero representa el volumen mensual de reseñas, y el segundo muestra la evolución del sentimiento promedio (calculado con VADER) por mes.",
        {
            "Período de análisis": "Mensual",
            "Método de análisis de sentimiento": "VADER"
        },
        data=monthly_stats
    )
    
    # Análisis por día de la semana
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    day_stats = df.groupby('day_of_week').agg({
        'review_id': 'count',
        'vader_compound': 'mean'
    }).reindex(range(7))
    day_stats.index = days
    day_stats.reset_index(inplace=True)
    day_stats.rename(columns={'index': 'Día'}, inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(day_stats['Día'], day_stats['review_id'], alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(day_stats['Día'], day_stats['vader_compound'], marker='o', color='red', linewidth=2)
    
    ax.set_title('Actividad y sentimiento por día de la semana')
    ax.set_ylabel('Número de reseñas')
    ax2.set_ylabel('Sentimiento promedio')
    
    plt.tight_layout()
    plt.savefig('actividad_semanal.png')
    plt.close()
    
    # Generar archivo de texto para actividad_semanal.png
    generate_text_for_png(
        'actividad_semanal',
        "Gráfico combinado que muestra la actividad y el sentimiento promedio de las reseñas por día de la semana. Las barras indican el número de reseñas, mientras que la línea roja muestra el sentimiento promedio (calculado con VADER).",
        {
            "Período de análisis": "Semanal (por día)",
            "Método de análisis de sentimiento": "VADER"
        },
        data=day_stats
    )
    
    return monthly_stats

# Ejecución principal
if __name__ == "__main__":
    # Parámetros configurables
    file_name = 'reviews5.csv'
    max_reviews = 50000
    
    # Cargar datos para análisis preliminar
    df_temp = load_data(file_name)
    if len(df_temp) > max_reviews:
        df_temp = df_temp.head(max_reviews)
    
    # Preprocesar para encontrar K óptimo
    print("\nRealizando preprocesamiento inicial para encontrar K óptimo...")
    nlp = spacy.load('en_core_web_sm')
    df_temp['preprocessed'] = df_temp['comments'].apply(
        lambda x: preprocess_text(x, nlp) if isinstance(x, str) else ''
    )
    
    # Vectorización
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=3)
    X_temp = vectorizer.fit_transform(df_temp['preprocessed']).toarray()
    
    # Encontrar K óptimo
    print("\nBuscando número óptimo de clusters...")
    k_optimal = 3
    print(f"\nSe utilizará K={k_optimal} para el análisis")
    
    # Ejecutar análisis completo
    df_result, user_data, report_data = analyze_reviews(
        file_name, 
        max_reviews=max_reviews,
        k_clusters=k_optimal,
        analyze_users=True
    )
    
    # Análisis de tendencias temporales
    temporal_data = analyze_temporal_trends(df_result)
    
    print("\nAnálisis completado exitosamente!")