# Descubrimientos Clave: EDA con Técnicas Estadísticas

## Pruebas de Normalidad
Los precios de Airbnb en las ciudades analizadas en 2024 muestran distribuciones complejas, con predominio de valores moderados pero influenciadas por precios extremos que incrementan la media y la dispersión

## Prueba de Kruskal-Wallis
El análisis mediante la prueba de Kruskal-Wallis reveló diferencias estadísticamente significativas (p < 0.05) en las medianas de los precios de alojamientos entre barrios de Madrid, Barcelona, Sevilla y Valencia, con Madrid exhibiendo la mayor heterogeneidad (estadístico = 14026.63), seguida por Barcelona, Valencia y Sevilla, lo que sugiere que la ubicación geográfica dentro de cada ciudad influye significativamente en los precios.

## Análisis de correlación
El análisis de correlación, previo al modelado predictivo de precios de alojamientos en Airbnb, identifica que las variables estructurales (capacidad, dormitorios y camas) exhiben correlaciones positivas moderadas a fuertes (Spearman hasta 0.631 en Madrid y Barcelona), siendo los predictores más relevantes para el precio, mientras que variables operativas como la tasa de respuesta o aceptación del anfitrión muestran asociaciones débiles, destacando una relación monótona negativa entre noches mínimas y precio (Spearman -0.65 en Madrid), lo que sugiere su inclusión en modelos predictivos para capturar dinámicas no lineales.

## Prueba de Mann-Whitney U	
La prueba de Mann-Whitney U, aplicada previo al modelado predictivo de precios de alojamientos en Airbnb, revela diferencias estadísticamente significativas (p < 0.05) en las distribuciones de precios, número de huéspedes, camas, dormitorios y baños entre alojamientos cercanos y alejados del centro turístico en Madrid, Barcelona, Sevilla y Valencia, salvo para baños en Valencia (p > 0.05), destacando la ubicación relativa al centro como un predictor clave para el precio debido a su influencia consistente en las variables estructurales.

## Intervalo de confianza para la media	
El análisis de intervalos de confianza del 95% para el precio promedio de alojamientos en Airbnb, previo al modelado predictivo, demuestra diferencias estadísticamente significativas entre ciudades (Sevilla: [180.60, 191.29] euros, Barcelona: [149.91, 153.78] euros, Madrid: [132.14, 136.25] euros, Valencia: [114.58, 117.87] euros), con intervalos no superpuestos que confirman la ciudad como un predictor crucial de precios, influido por factores como demanda turística y variabilidad local, siendo Sevilla la de mayor precio medio y Valencia la de menor.

## Análisis de clustering espacial
El análisis de clustering espacial con DBSCAN, previo al modelado predictivo de precios de alojamientos en Airbnb, revela una segmentación geográfica heterogénea en Madrid, Barcelona, Sevilla y Valencia, con clústeres principales que agrupan la mayoría de las propiedades (Madrid: media 134,05 euros, Sevilla: media más alta) y clústeres menores con precios diferenciados (desde 57 a 500 euros), destacando la coexistencia de submercados de lujo y económicos en áreas cercanas, lo que subraya la ubicación y la densidad espacial como predictores clave para los precios.

## Análisis de autocorrelación espacial
El análisis del índice de Moran, previo al modelado predictivo de precios de alojamientos en Airbnb, revela una autocorrelación espacial positiva moderada (índices entre 0.35 y 0.41, p < 0.001) en Madrid, Barcelona, Sevilla y Valencia, con Valencia mostrando la mayor agrupación geográfica de precios similares (0.41), lo que indica que la proximidad espacial es un predictor clave para los precios debido a la tendencia de propiedades con valores similares a concentrarse geográficamente.

## Prueba de Chi-cuadrado de independencia	
La prueba de Chi-cuadrado de independencia, aplicada previo al modelado predictivo de precios de alojamientos en Airbnb, demuestra una dependencia estadísticamente significativa (p < 0.05) entre vecindario, tipo de alojamiento y tipo de habitación en Madrid, Barcelona, Sevilla y Valencia, con estadísticos elevados (e.g., Madrid: 131210 para vecindario-tipo de alojamiento; Barcelona: 184938 para tipo de alojamiento interno), destacando que la distribución geográfica y categórica de las propiedades es un predictor clave para los precios debido a la heterogeneidad espacial y estructural de la oferta.

## Análisis de correspondencias	
El análisis de correspondencias, previo al modelado predictivo de precios de alojamientos en Airbnb, revela asociaciones significativas entre barrios, precios y número de amenidades en Madrid, Barcelona, Sevilla y Valencia, destacando una segmentación espacial donde barrios céntricos y turísticos (e.g., Recoletos en Madrid, Santa Cruz en Sevilla) se asocian con precios altos, mientras que zonas periféricas (e.g., Puerta del Ángel en Madrid, Nou Moles en Valencia) se vinculan a precios bajos, con amenidades actuando como diferenciador competitivo en algunos casos, lo que subraya la localización y equipamiento como predictores clave para los precios.

## Análisis de componentes principales	
El análisis de componentes principales , previo al modelado predictivo de precios de alojamientos en Airbnb, revela que en Madrid, Barcelona, Sevilla y Valencia, la variabilidad de los datos se explica principalmente por dos dimensiones: la calidad percibida (valoraciones de limpieza y comunicación, 24% de varianza en Madrid) y las características estructurales (camas, dormitorios, baños, con cargas altas de precio, e.g., 0.295 en Barcelona), destacando que el precio se alinea más con el tamaño en Madrid y Barcelona, con la actividad del anfitrión en Sevilla, y con estructura y percepción de localización en Valencia, posicionando estas variables como predictores clave para los precios.

## Prueba exacta de Fisher	
La prueba exacta de Fisher, aplicada previo al modelado predictivo de precios de alojamientos en Airbnb, demuestra una asociación significativa (p < 0.05) entre el número de amenidades y el nivel de precio en Madrid, Barcelona, Sevilla y Valencia, con odds ratios que indican una mayor probabilidad de precios altos en alojamientos con más amenidades (Barcelona: 2.149, Madrid: 1.7002, Valencia: 1.2993, Sevilla: 1.0598), destacando las amenidades como un predictor clave para los precios, especialmente en mercados competitivos como Barcelona.

## Tamaño del efecto mediante el estadístico Cohen’s d	
El análisis del tamaño del efecto mediante Cohen’s d, previo al modelado predictivo de precios de alojamientos en Airbnb, revela que la capacidad, el tipo de propiedad y la ubicación son predictores clave de los precios en Barcelona, Madrid, Sevilla y Valencia, con efectos grandes (Cohen’s d ≥ 0.8) como el de -4.0630 en Barcelona entre poca capacidad y habitaciones compartidas, o -3.5581 en Valencia entre poca capacidad y el barrio de Carpesa, destacando que alojamientos más grandes, propiedades exclusivas (e.g., villas, castillos) y ubicaciones premium generan diferencias de precio significativas.


## Test de Friedman	
El hallazgo del test de Friedman sugiere que, para el modelado predictivo de precios, las combinaciones de algunas características físicas (dormitorios, camas, baños, capacidad) no deberían ser tratadas como un bloque único o interdependiente dentro de los barrios analizados, ya que no generan diferencias significativas en los precios. Sin embargo, esto no descarta la importancia de estas variables de forma individual o en otros contextos, como lo indican los hallazgos previos.
