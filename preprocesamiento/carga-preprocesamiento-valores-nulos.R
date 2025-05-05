################################################################################
################################################################################
# cÓDIGO BASE CARGA, PREPROCESAMIENTO Y LIMPIEZA DE LOS DATOS #
# PEC3 - ÁNGEL SOTO GARCÍA - 22.536 TRABAJO FINAL DE GRADO

################################################################################
################################################################################
# LIBRERÍAS
options(warn = -1)
if(!require(naniar)) install.packages("naniar")
if(!require(mice)) install.packages("mice") 
if(!require(naniar)) install.packages("FNN")
if(!require(mice)) install.packages("VIM") 
if(!require(naniar)) install.packages("scales")
if(!require(mice)) install.packages("tidyr") 
if(!require(naniar)) install.packages("dplyr")
if(!require(mice)) install.packages("ggplot2") 
if(!require(naniar)) install.packages("RPostgres")

library(RPostgres)
library(mice)
library(FNN)
library(VIM)
library(scales)
library(tidyr)
library(dplyr)
library(ggplot2)
library(RPostgres)
################################################################################
################################################################################
# FUNCIÓN PARA TRANSFORMAR LOS TIPOS DE DATOS EN EL DATAFRAME DE PROPIEDADES
transform_property_dataframe <- function(df) {
  df <- df %>%
    mutate(
      id = as.integer(id),scrape_id = as.integer(scrape_id), host_id = as.integer(host_id),
      host_listings_count = as.integer(host_listings_count),
      host_total_listings_count = as.integer(host_total_listings_count),
      accommodates = as.integer(accommodates),
      bathrooms = as.integer(bathrooms),
      bedrooms = as.integer(bedrooms),
      beds = as.integer(beds),
      minimum_nights = as.integer(minimum_nights),
      maximum_nights = as.integer(maximum_nights),
      minimum_minimum_nights = as.integer(minimum_minimum_nights),
      maximum_minimum_nights = as.integer(maximum_minimum_nights),
      minimum_maximum_nights = as.integer(minimum_maximum_nights),
      maximum_maximum_nights = as.integer(maximum_maximum_nights),
      availability_30 = as.integer(availability_30),
      availability_60 = as.integer(availability_60),
      availability_90 = as.integer(availability_90),
      availability_365 = as.integer(availability_365),
      number_of_reviews = as.integer(number_of_reviews),
      number_of_reviews_ltm = as.integer(number_of_reviews_ltm),
      number_of_reviews_l30d = as.integer(number_of_reviews_l30d),
      calculated_host_listings_count = as.integer(calculated_host_listings_count),
      calculated_host_listings_count_entire_homes = as.integer(calculated_host_listings_count_entire_homes),
      calculated_host_listings_count_private_rooms = as.integer(calculated_host_listings_count_private_rooms),
      calculated_host_listings_count_shared_rooms = as.integer(calculated_host_listings_count_shared_rooms)
    )

  df <- df %>%
    mutate(
      latitude = as.numeric(latitude),
      longitude = as.numeric(longitude),
      price = as.numeric(gsub("[\\$,]", "", price)),
      minimum_nights_avg_ntm = as.numeric(minimum_nights_avg_ntm),
      maximum_nights_avg_ntm = as.numeric(maximum_nights_avg_ntm),
      review_scores_rating = as.numeric(review_scores_rating),
      review_scores_accuracy = as.numeric(review_scores_accuracy),
      review_scores_cleanliness = as.numeric(review_scores_cleanliness),
      review_scores_checkin = as.numeric(review_scores_checkin),
      review_scores_communication = as.numeric(review_scores_communication),
      review_scores_location = as.numeric(review_scores_location),
      review_scores_value = as.numeric(review_scores_value),
      reviews_per_month = as.numeric(reviews_per_month),
      host_response_rate = as.numeric(gsub("[\\%]", "", host_response_rate)),
      host_acceptance_rate = as.numeric(gsub("[\\%]", "", host_acceptance_rate))
    )
  
  # Transformar columnas a fechas (Date)
  df <- df %>%
    mutate(
      last_scraped = as.Date(last_scraped),
      host_since = as.Date(host_since),
      first_review = as.Date(first_review),
      last_review = as.Date(last_review),
      calendar_last_scraped = as.Date(calendar_last_scraped)
    )
  
  # Transformar columnas a lógicas (logical)
  df <- df %>%
    mutate(
      host_is_superhost = as.logical(host_is_superhost),
      host_has_profile_pic = as.logical(host_has_profile_pic),
      host_identity_verified = as.logical(host_identity_verified),
      has_availability = as.logical(has_availability),
      instant_bookable = as.logical(instant_bookable)
    )
  
  # Transformar columnas a categóricas (factor)
  df <- df %>%
    mutate(
      host_response_time = as.factor(host_response_time),
      host_neighbourhood = as.factor(host_neighbourhood),
      neighbourhood = as.factor(neighbourhood),
      neighbourhood_cleansed = as.factor(neighbourhood_cleansed),
      neighbourhood_group_cleansed = as.factor(neighbourhood_group_cleansed),
      property_type = as.factor(property_type),
      room_type = as.factor(room_type)
    )
  
  # Sustituir valores nulos en campos específicos
  if("id" %in% names(df)) {
    df <- df %>% mutate(id = ifelse(is.na(id), 0000, id))
  }
  
  if("license" %in% names(df)) {
    df <- df %>% mutate(license = ifelse(is.na(license), "No disponible", license))
  }
  
  return(df)
}

# Función para transformar los tipos de datos en el dataframe de reviews
transform_reviews_dataframe <- function(df) {
  df <- df %>%
    mutate(
      listing_id = as.integer(listing_id),
      id = as.integer(id),
      reviewer_id = as.integer(reviewer_id),
      date = as.Date(date)
    )
  
  # Sustituir valores nulos en campos específicos
  if("id" %in% names(df)) {
    df <- df %>% mutate(id = ifelse(is.na(id), 0000, id))
  }
  
  return(df)
}
################################################################################
################################################################################
# FUNCIÓN PARA CREAR GRÁFICO VALORES NULOS EN / CAMPO 
create_na_plot <- function(df, color, city_name) {
  total_registros <- nrow(df)
  na_analysis <- df %>%
    summarise(across(everything(), ~sum(is.na(.)))) %>%
    pivot_longer(everything(), names_to = "Variable", values_to = "NAs") %>%
    filter(NAs > 0)
  

  max_na <- max(na_analysis$NAs, na.rm = TRUE)
  margin_right <- max_na * 1.2
  
  ggplot(na_analysis, aes(x = reorder(Variable, NAs), y = NAs)) +
    geom_col(fill = color, alpha = 0.9, width = 0.7) +
    geom_text(aes(label = comma(NAs)), hjust = -0.1, size = 3.5, color = "#34495e") +
    coord_flip(ylim = c(0, margin_right)) +
    labs(title = "DISTRIBUCIÓN DE VALORES NULOS",
         subtitle = paste("Análisis de datos faltantes en", city_name, "- Total registros:", format(total_registros, big.mark = ",")),
         x = "", y = "") +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, color = "#2c3e50"),
      plot.subtitle = element_text(hjust = 0.5, color = "#7f8c8d", margin = margin(b = 20)),
      panel.grid = element_blank(),
      axis.text.x = element_blank(),
      axis.text.y = element_text(size = 11, color = "#2c3e50"),
      plot.margin = unit(c(1, 3, 1, 1), "cm")
    )
}



# Función para eliminar columnas con 90% o más de nulidad y limpiar registros según umbral de nulidad
clean_dataframe <- function(df, threshold_percentage = 5) {
  total_rows <- nrow(df)
  threshold_count <- ceiling((threshold_percentage / 100) * total_rows)
  
  # Identificar columnas con 90% o más de valores nulos
  cols_high_na <- names(df)[colSums(is.na(df)) >= 0.9 * total_rows]
  
  # Identificar columnas con menos del umbral de valores nulos (5% o menos)
  cols_few_na <- names(df)[colSums(is.na(df)) > 0 & colSums(is.na(df)) <= threshold_count]
  
  # Eliminar columnas con 90% o más de valores nulos
  if (length(cols_high_na) > 0) {
    df <- df %>% select(-all_of(cols_high_na))
    message(paste("Eliminadas", length(cols_high_na), "columnas con ≥90% de valores nulos:", 
                  paste(cols_high_na, collapse=", ")))
  }
  
  # Para las columnas con pocos NA (5% o menos), eliminar las filas con valores nulos
  if (length(cols_few_na) > 0) {
    rows_before <- nrow(df)
    
    # Crear una condición para las filas a mantener (aquellas sin NA en las columnas identificadas)
    condition <- rowSums(is.na(df[cols_few_na])) == 0
    df <- df[condition, ]
    
    rows_removed <- rows_before - nrow(df)
    message(paste("Eliminadas", rows_removed, "filas con valores nulos en columnas con ≤", 
                  threshold_percentage, "% de nulidad"))
  }
  
  return(df)
}
################################################################################
################################################################################
# FUNCIÓN PARA LIMPIEZA INICIAL DE VALORES NULOS
clean_reviews_dataframe <- function(df) {
  # Eliminar la columna 'id'
  df <- df %>% select(-id)
  
  # Eliminar filas con valores nulos en 'listing_id'
  rows_before <- nrow(df)
  df <- df %>% filter(!is.na(listing_id))
  rows_removed <- rows_before - nrow(df)
  
  message(paste("Eliminada columna 'id' y", rows_removed, "filas con valores nulos en 'listing_id'"))
  
  return(df)
}
################################################################################
################################################################################
# CONEXIÓN A LA BASE DE DATOS
con <- dbConnect(RPostgres::Postgres(), 
                 dbname = "postgres", 
                 host = "localhost", 
                 port = 5434, 
                 user = "postgres", 
                 password = "Dum213dum")
################################################################################
################################################################################
# CARGA DE LOS DATOS Y VISUALIZACION DE VALORES NULOS
df_madrid <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_madrid;")
df_madrid <- transform_property_dataframe(df_madrid)
print(create_na_plot(df_madrid, "#C41E3A", "Madrid"))

df_euskadi <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_euskadi;")
df_euskadi <- transform_property_dataframe(df_euskadi)
print(create_na_plot(df_euskadi, "#008C45", "Euskadi"))

df_girona <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_girona;")
df_girona <- transform_property_dataframe(df_girona)
print(create_na_plot(df_girona, "#DA121A", "Girona"))

df_malaga <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_malaga;")
df_malaga <- transform_property_dataframe(df_malaga)
print(create_na_plot(df_malaga, "#00579C", "Málaga"))

df_barcelona <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_barcelona;")
df_barcelona <- transform_property_dataframe(df_barcelona)
print(create_na_plot(df_barcelona, "#00579C", "Barcelona"))


df_mallorca <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_mallorca;")
df_mallorca <- transform_property_dataframe(df_mallorca)
print(create_na_plot(df_mallorca, "#F9A01B", "Mallorca"))

df_sevilla <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_sevilla;")
df_sevilla <- transform_property_dataframe(df_sevilla)
print(create_na_plot(df_sevilla, "#8F001A", "Sevilla"))

df_valencia <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_valencia;")
df_valencia <- transform_property_dataframe(df_valencia)
print(create_na_plot(df_valencia, "#EB2727", "Valencia"))

df_menorca <- dbGetQuery(con, "SELECT * FROM airnb.propiedades_menorca;")
df_menorca <- transform_property_dataframe(df_menorca)
print(create_na_plot(df_menorca, "#7BB6B1", "Menorca"))


df_reviews <- dbGetQuery(con, "SELECT * FROM airnb.reviews;")
df_reviews <- transform_reviews_dataframe(df_reviews)
print(create_na_plot(df_reviews, "#3498db", "reseñas"))
################################################################################
################################################################################
# INFORMACIÓN INICIAL VALORES NULOS PROPIEDADES

cities <- c("madrid", "euskadi", "girona", "malaga", "mallorca", "sevilla",
            "valencia", "menorca","barcelona")
summary_table <- data.frame(
  Ciudad = character(),
  Total_Registros = integer(),
  Campos_con_NA = integer(),
  Porcentaje_Completitud = numeric(),
  stringsAsFactors = FALSE
)

for (city in cities) {
  df_name <- paste0("df_", city)
  df <- get(df_name)
  total_registros <- nrow(df)
  total_campos <- ncol(df)
  campos_con_na <- sum(colSums(is.na(df)) > 0)
  total_celdas <- total_registros * total_campos
  celdas_no_na <- sum(!is.na(df))
  porcentaje_completitud <- round(celdas_no_na / total_celdas * 100, 2)
  summary_table <- rbind(summary_table, 
                         data.frame(Ciudad = city, 
                                    Total_Registros = total_registros,
                                    Campos_con_NA = campos_con_na,
                                    Porcentaje_Completitud = porcentaje_completitud))
}

print(summary_table)
################################################################################
################################################################################
# INFORMACIÓN INICIAL VALORES NULOS RESEÑAS
total_registros <- nrow(df_reviews)
total_campos <- ncol(df_reviews)
campos_con_na <- sum(colSums(is.na(df_reviews)) > 0)
celdas_no_na <- sum(!is.na(df_reviews))
porcentaje_completitud <- round(celdas_no_na / (total_registros * total_campos) * 100, 2)
summary_table <- rbind(summary_table, 
                       data.frame(Ciudad = "reviews", 
                                  Total_Registros = total_registros,
                                  Campos_con_NA = campos_con_na,
                                  Porcentaje_Completitud = porcentaje_completitud))

print(summary_table)
################################################################################
################################################################################
# LIMPIEZA DE VALORES NULOS EN PROPIEDADES
show_column_types <- function(df, num_cols = 10) {
  col_types <- sapply(df[, 1:min(num_cols, ncol(df))], class)
  col_types_df <- data.frame(
    Columna = names(col_types),
    Tipo = sapply(col_types, function(x) x[1])
  )
  return(col_types_df)
}

cleaning_results <- data.frame(
  Dataset = character(),
  Registros_Antes = integer(),
  Registros_Despues = integer(),
  Columnas_Antes = integer(),
  Columnas_Despues = integer(),
  stringsAsFactors = FALSE
)

log_cleaning_results <- function(name, df_before, df_after) {
  cleaning_results <<- rbind(cleaning_results,
                             data.frame(
                               Dataset = name,
                               Registros_Antes = nrow(df_before),
                               Registros_Despues = nrow(df_after),
                               Columnas_Antes = ncol(df_before),
                               Columnas_Despues = ncol(df_after),
                               stringsAsFactors = FALSE
                             ))
}

for (city in cities) {
  df_name <- paste0("df_", city)
  df_before <- get(df_name)
  cat(paste("\n\n--- Limpieza de", toupper(city), "---\n"))
  df_after <- clean_dataframe(df_before, threshold_percentage = 5)
  assign(df_name, df_after)
  log_cleaning_results(city, df_before, df_after)
}

################################################################################
################################################################################
# LIMPIEZA DE VALORES NULOS EN RESEÑAS
df_reviews_before <- df_reviews
df_reviews <- clean_reviews_dataframe(df_reviews)
log_cleaning_results("reviews", df_reviews_before, df_reviews)
################################################################################
################################################################################
imputation_results <- data.frame(
  Dataset = character(),
  NAs_Antes = integer(),
  NAs_Despues = integer(),
  Tiempo_Ejecucion = numeric(),
  stringsAsFactors = FALSE
)

log_imputation_results <- function(name, df_before, df_after, start_time) {
  imputation_results <<- rbind(imputation_results,
                               data.frame(
                                 Dataset = name,
                                 NAs_Antes = sum(is.na(df_before)),
                                 NAs_Despues = sum(is.na(df_after)),
                                 Tiempo_Ejecucion = round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2),
                                 stringsAsFactors = FALSE
                               ))
}



verification_results <- data.frame(
  Dataset = character(),
  Total_Registros = integer(),
  Total_Campos = integer(),
  NAs_Restantes = integer(),
  Completitud = character(),
  stringsAsFactors = FALSE
)

print(verification_results)
################################################################################
################################################################################
# EUSKADI (imputacion final de valores nulos)

print(create_na_plot(df_euskadi, "#C41E3A", "euskadi"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price","review_scores_value","review_scores_location","review_scores_communication","review_scores_cleanliness",
  "review_scores_checkin", "review_scores_accuracy", "reviews_per_month", "review_scores_rating")


df_mice_numeric <- df_euskadi[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)
print(resultado)


methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars


imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)


df_euskadi_imputed_numeric <- complete(imputed_numeric, 1)



df_euskadi[, numeric_vars] <- df_euskadi_imputed_numeric[, numeric_vars]
print(create_na_plot(df_euskadi, "#C41E3A", "euskadi"))


categorical_vars <- c(
  "host_location",
  "neighborhood_overview",
  "neighbourhood","first_review","last_review"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_euskadi)

for (var in categorical_vars) {
  if (var %in% names(df_euskadi)) {
    na_count <- sum(is.na(df_euskadi[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_euskadi[[var]] <- as.character(df_euskadi[[var]])
      df_euskadi[[var]][is.na(df_euskadi[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_euskadi[[var]])) > 1 || na_count < threshold) {
        df_euskadi[[var]] <- as.factor(df_euskadi[[var]])
      } else {
        df_euskadi[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}
################################################################################
################################################################################
# MENORCA (imputación de valores nulos)

print(create_na_plot(df_menorca, "#C41E3A", "menorca"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price", "minimum_nights", "maximum_nights",
  "minimum_minimum_nights", "maximum_minimum_nights",
  "minimum_maximum_nights", "maximum_maximum_nights",
  "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
  "availability_30", "availability_60", "availability_90",
  "availability_365", "number_of_reviews", "number_of_reviews_ltm",
  "number_of_reviews_l30d", "review_scores_rating", "review_scores_accuracy",
  "review_scores_cleanliness", "review_scores_checkin",
  "review_scores_communication", "review_scores_location",
  "review_scores_value", "calculated_host_listings_count",
  "calculated_host_listings_count_entire_homes",
  "calculated_host_listings_count_private_rooms",
  "calculated_host_listings_count_shared_rooms", "reviews_per_month"
)

df_mice_numeric <- df_menorca[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)

print(resultado)

methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars


imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)


df_menorca_imputed_numeric <- complete(imputed_numeric, 1)

df_menorca[, numeric_vars] <- df_menorca_imputed_numeric[, numeric_vars]
print(create_na_plot(df_menorca, "#C41E3A", "menorca"))


categorical_vars <- c(
  "host_location",
  "neighborhood_overview",
  "neighbourhood", "first_review", "last_review","host_neighbourhood"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_menorca)

for (var in categorical_vars) {
  if (var %in% names(df_menorca)) {
    na_count <- sum(is.na(df_menorca[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_menorca[[var]] <- as.character(df_menorca[[var]])
      df_menorca[[var]][is.na(df_menorca[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_menorca[[var]])) > 1 || na_count < threshold) {
        df_menorca[[var]] <- as.factor(df_menorca[[var]])
      } else {
        df_menorca[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}

################################################################################
################################################################################
# MALLORCA (imputación de valores nulos)

print(create_na_plot(df_mallorca, "#C41E3A", "mallorca"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price", "minimum_nights", "maximum_nights",
  "minimum_minimum_nights", "maximum_minimum_nights",
  "minimum_maximum_nights", "maximum_maximum_nights",
  "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
  "availability_30", "availability_60", "availability_90",
  "availability_365", "number_of_reviews", "number_of_reviews_ltm",
  "number_of_reviews_l30d", "review_scores_rating", "review_scores_accuracy",
  "review_scores_cleanliness", "review_scores_checkin",
  "review_scores_communication", "review_scores_location",
  "review_scores_value", "calculated_host_listings_count",
  "calculated_host_listings_count_entire_homes",
  "calculated_host_listings_count_private_rooms",
  "calculated_host_listings_count_shared_rooms", "reviews_per_month"
)

df_mice_numeric <- df_mallorca[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)


print(resultado)


methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars


imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)


df_mallorca_imputed_numeric <- complete(imputed_numeric, 1)


df_mallorca[, numeric_vars] <- df_mallorca_imputed_numeric[, numeric_vars]
print(create_na_plot(df_mallorca, "#C41E3A", "mallorca"))


categorical_vars <- c(
  "host_location","host_neighbourhood",
  "neighborhood_overview",
  "neighbourhood", "first_review", "last_review"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_mallorca)

for (var in categorical_vars) {
  if (var %in% names(df_mallorca)) {
    na_count <- sum(is.na(df_mallorca[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_mallorca[[var]] <- as.character(df_mallorca[[var]])
      df_mallorca[[var]][is.na(df_mallorca[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_mallorca[[var]])) > 1 || na_count < threshold) {
        df_mallorca[[var]] <- as.factor(df_mallorca[[var]])
      } else {
        df_mallorca[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}

################################################################################
################################################################################
# GIRONA (imputación de valores nulos)

print(create_na_plot(df_girona, "#C41E3A", "girona"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price", "minimum_nights", "maximum_nights",
  "minimum_minimum_nights", "maximum_minimum_nights",
  "minimum_maximum_nights", "maximum_maximum_nights",
  "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
  "availability_30", "availability_60", "availability_90",
  "availability_365", "number_of_reviews", "number_of_reviews_ltm",
  "number_of_reviews_l30d", "review_scores_rating", "review_scores_accuracy",
  "review_scores_cleanliness", "review_scores_checkin",
  "review_scores_communication", "review_scores_location",
  "review_scores_value", "calculated_host_listings_count",
  "calculated_host_listings_count_entire_homes",
  "calculated_host_listings_count_private_rooms",
  "calculated_host_listings_count_shared_rooms", "reviews_per_month"
)

df_mice_numeric <- df_girona[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)


print(resultado)


methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars


imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)


df_girona_imputed_numeric <- complete(imputed_numeric, 1)


df_girona[, numeric_vars] <- df_girona_imputed_numeric[, numeric_vars]
print(create_na_plot(df_girona, "#C41E3A", "girona"))


categorical_vars <- c(
  "host_location",
  "neighborhood_overview",
  "neighbourhood", "first_review", "last_review"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_girona)

for (var in categorical_vars) {
  if (var %in% names(df_girona)) {
    na_count <- sum(is.na(df_girona[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_girona[[var]] <- as.character(df_girona[[var]])
      df_girona[[var]][is.na(df_girona[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_girona[[var]])) > 1 || na_count < threshold) {
        df_girona[[var]] <- as.factor(df_girona[[var]])
      } else {
        df_girona[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}

################################################################################
################################################################################
# SEVILLA (imputación de valores nulos)

print(create_na_plot(df_sevilla, "#C41E3A", "sevilla"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price", "minimum_nights", "maximum_nights",
  "minimum_minimum_nights", "maximum_minimum_nights",
  "minimum_maximum_nights", "maximum_maximum_nights",
  "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
  "availability_30", "availability_60", "availability_90",
  "availability_365", "number_of_reviews", "number_of_reviews_ltm",
  "number_of_reviews_l30d", "review_scores_rating", "review_scores_accuracy",
  "review_scores_cleanliness", "review_scores_checkin",
  "review_scores_communication", "review_scores_location",
  "review_scores_value", "calculated_host_listings_count",
  "calculated_host_listings_count_entire_homes",
  "calculated_host_listings_count_private_rooms",
  "calculated_host_listings_count_shared_rooms", "reviews_per_month"
)

df_mice_numeric <- df_sevilla[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)

print(resultado)


methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars


imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)


df_sevilla_imputed_numeric <- complete(imputed_numeric, 1)


df_sevilla[, numeric_vars] <- df_sevilla_imputed_numeric[, numeric_vars]
print(create_na_plot(df_sevilla, "#C41E3A", "sevilla"))


categorical_vars <- c(
  "host_location","host_neighbourhood",
  "neighborhood_overview",
  "neighbourhood", "first_review", "last_review"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_sevilla)

for (var in categorical_vars) {
  if (var %in% names(df_sevilla)) {
    na_count <- sum(is.na(df_sevilla[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_sevilla[[var]] <- as.character(df_sevilla[[var]])
      df_sevilla[[var]][is.na(df_sevilla[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_sevilla[[var]])) > 1 || na_count < threshold) {
        df_sevilla[[var]] <- as.factor(df_sevilla[[var]])
      } else {
        df_sevilla[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}
################################################################################
################################################################################
# VALENCIA (imputación de valores nulos)

print(create_na_plot(df_valencia, "#C41E3A", "valencia"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price", "minimum_nights", "maximum_nights",
  "minimum_minimum_nights", "maximum_minimum_nights",
  "minimum_maximum_nights", "maximum_maximum_nights",
  "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
  "availability_30", "availability_60", "availability_90",
  "availability_365", "number_of_reviews", "number_of_reviews_ltm",
  "number_of_reviews_l30d", "review_scores_rating", "review_scores_accuracy",
  "review_scores_cleanliness", "review_scores_checkin",
  "review_scores_communication", "review_scores_location",
  "review_scores_value", "calculated_host_listings_count",
  "calculated_host_listings_count_entire_homes",
  "calculated_host_listings_count_private_rooms",
  "calculated_host_listings_count_shared_rooms", "reviews_per_month"
)

df_mice_numeric <- df_valencia[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)

# Imprimir el resultado de la prueba
print(resultado)

# Configurar métodos solo para numéricas
methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars

# Imputar numéricas
imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)

# Obtener datos imputados
df_valencia_imputed_numeric <- complete(imputed_numeric, 1)

# Combinar resultados
df_valencia[, numeric_vars] <- df_valencia_imputed_numeric[, numeric_vars]
print(create_na_plot(df_valencia, "#C41E3A", "valencia"))

# 2. Sustitución de nulos en campos categóricos importantes
categorical_vars <- c(
  "host_location",
  "neighborhood_overview",
  "neighbourhood", "first_review", "last_review", "host_neighbourhood"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_valencia)

for (var in categorical_vars) {
  if (var %in% names(df_valencia)) {
    na_count <- sum(is.na(df_valencia[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_valencia[[var]] <- as.character(df_valencia[[var]])
      df_valencia[[var]][is.na(df_valencia[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_valencia[[var]])) > 1 || na_count < threshold) {
        df_valencia[[var]] <- as.factor(df_valencia[[var]])
      } else {
        df_valencia[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}

################################################################################
################################################################################
# MÁLAGA (imputación de valores nulos)

print(create_na_plot(df_malaga, "#C41E3A", "malaga"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price", "minimum_nights", "maximum_nights",
  "minimum_minimum_nights", "maximum_minimum_nights",
  "minimum_maximum_nights", "maximum_maximum_nights",
  "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
  "availability_30", "availability_60", "availability_90",
  "availability_365", "number_of_reviews", "number_of_reviews_ltm",
  "number_of_reviews_l30d", "review_scores_rating", "review_scores_accuracy",
  "review_scores_cleanliness", "review_scores_checkin",
  "review_scores_communication", "review_scores_location",
  "review_scores_value", "calculated_host_listings_count",
  "calculated_host_listings_count_entire_homes",
  "calculated_host_listings_count_private_rooms",
  "calculated_host_listings_count_shared_rooms", "reviews_per_month"
)

df_mice_numeric <- df_malaga[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)

print(resultado)


methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars


imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)


df_malaga_imputed_numeric <- complete(imputed_numeric, 1)


df_malaga[, numeric_vars] <- df_malaga_imputed_numeric[, numeric_vars]
print(create_na_plot(df_malaga, "#C41E3A", "malaga"))


categorical_vars <- c(
  "host_location",
  "neighborhood_overview",
  "neighbourhood", "first_review", "last_review", "host_neighbourhood"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_malaga)

for (var in categorical_vars) {
  if (var %in% names(df_malaga)) {
    na_count <- sum(is.na(df_malaga[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_malaga[[var]] <- as.character(df_malaga[[var]])
      df_malaga[[var]][is.na(df_malaga[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_malaga[[var]])) > 1 || na_count < threshold) {
        df_malaga[[var]] <- as.factor(df_malaga[[var]])
      } else {
        df_malaga[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}

################################################################################
################################################################################
# BARCELONA (imputación de valores nulos)

print(create_na_plot(df_barcelona, "#C41E3A", "barcelona"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price", "minimum_nights", "maximum_nights", "review_scores_rating", "review_scores_accuracy",
  "review_scores_cleanliness", "review_scores_checkin",
  "review_scores_communication", "review_scores_location",
  "review_scores_value","reviews_per_month"
)

df_mice_numeric <- df_barcelona[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)


print(resultado)


methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars


imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)


df_barcelona_imputed_numeric <- complete(imputed_numeric, 1)


df_barcelona[, numeric_vars] <- df_barcelona_imputed_numeric[, numeric_vars]
print(create_na_plot(df_barcelona, "#C41E3A", "barcelona"))


categorical_vars <- c(
  "host_location",
  "neighborhood_overview",
  "neighbourhood", "first_review", "last_review",  "host_neighbourhood"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_barcelona)

for (var in categorical_vars) {
  if (var %in% names(df_barcelona)) {
    na_count <- sum(is.na(df_barcelona[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_barcelona[[var]] <- as.character(df_barcelona[[var]])
      df_barcelona[[var]][is.na(df_barcelona[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_barcelona[[var]])) > 1 || na_count < threshold) {
        df_barcelona[[var]] <- as.factor(df_barcelona[[var]])
      } else {
        df_barcelona[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}

################################################################################
################################################################################
# MADRID (imputación de valores nulos )

print(create_na_plot(df_madrid, "#C41E3A", "madrid"))

numeric_vars <- c(
  "host_response_rate", "host_acceptance_rate",
  "host_listings_count", "host_total_listings_count",
  "latitude", "longitude", "accommodates", "bathrooms",
  "bedrooms", "beds", "price", "minimum_nights", "maximum_nights", "review_scores_rating", "review_scores_accuracy",
  "review_scores_cleanliness", "review_scores_checkin",
  "review_scores_communication", "review_scores_location",
  "review_scores_value","reviews_per_month"
)

df_mice_numeric <- df_madrid[, numeric_vars]
resultado <- mcar_test(df_mice_numeric)


print(resultado)


methods_numeric <- rep("pmm", length(numeric_vars))
names(methods_numeric) <- numeric_vars


imputed_numeric <- mice(df_mice_numeric, 
                        method = methods_numeric,
                        m = 10,
                        maxit = 20,
                        seed = 123)


df_madrid_imputed_numeric <- complete(imputed_numeric, 1)


df_madrid[, numeric_vars] <- df_madrid_imputed_numeric[, numeric_vars]
print(create_na_plot(df_madrid, "#C41E3A", "madrid"))


categorical_vars <- c(
  "host_location",
  "neighborhood_overview",
  "neighbourhood", "first_review", "last_review"
)

# Umbral mínimo de registros para considerar la sustitución (1% del total)
threshold <- 0.01 * nrow(df_madrid)

for (var in categorical_vars) {
  if (var %in% names(df_madrid)) {
    na_count <- sum(is.na(df_madrid[[var]]))
    if (na_count > 0) {
      # Convertir a carácter y reemplazar NAs
      df_madrid[[var]] <- as.character(df_madrid[[var]])
      df_madrid[[var]][is.na(df_madrid[[var]])] <- "No disponible"
      
      # Convertir de vuelta a factor solo si hay suficientes valores distintos
      if (length(unique(df_madrid[[var]])) > 1 || na_count < threshold) {
        df_madrid[[var]] <- as.factor(df_madrid[[var]])
      } else {
        df_madrid[[var]] <- NULL  # Eliminar columna si solo tiene "No disponible"
      }
      
      message(paste("Procesada variable categórica:", var, "-", na_count, "NAs reemplazados"))
    }
  }
}
################################################################################
################################################################################
# SELECCIÓN DE CARACTERÍSTICAS
datasets <- list(
  "Euskadi" = df_euskadi,
  "Girona" = df_girona,
  "Málaga" = df_malaga,
  "Mallorca" = df_mallorca,
  "Sevilla" = df_sevilla,
  "Valencia" = df_valencia,
  "Menorca" = df_menorca,
  "Madrid" = df_madrid,
  "Barcelona" = df_barcelona
)

# Variables a seleccionar en cada dataframe
selected_vars <- c(
  "id",
  "last_scraped",
  "host_response_rate",
  "host_response_time",
  "host_acceptance_rate",
  "latitude",
  "longitude",
  "availability_365",
  "neighbourhood_cleansed",    # Ubicación
  "property_type",            # Características del inmueble
  "room_type",
  "accommodates",
  "bathrooms",
  "bedrooms",
  "beds",
  "amenities",                # Amenidades
  "host_since",               # Características del anfitrión
  "host_listings_count",
  "host_total_listings_count",
  "number_of_reviews",        # Reseñas
  "review_scores_rating",
  "review_scores_location",
  "review_scores_communication",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "minimum_nights",           # Restricciones de noches
  "maximum_nights",
  "price"                     # Variable objetivo
)

# Función para seleccionar las variables relevantes en cada dataframe
filter_datasets <- function(df_list, vars) {
  filtered_list <- lapply(df_list, function(df) {
    # Verificar qué variables existen en el dataframe actual
    existing_vars <- vars[vars %in% names(df)]
    # Seleccionar solo las variables existentes
    df_selected <- df[, existing_vars, drop = FALSE]
    return(df_selected)
  })
  return(filtered_list)
}

# Aplicar la función a la lista de datasets
filtered_datasets <- filter_datasets(datasets, selected_vars)


filtered_datasets[["Barcelona"]]