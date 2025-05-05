################################################################################
################################################################################
# ANÁLISIS MEJORADO DE VALORES ATÍPICOS
################################################################################
################################################################################
# LIBRERÍAS
library(dplyr)
library(ggplot2)
library(gridExtra)
library(scales)   
library(tidyr)    
################################################################################
################################################################################
# DEFINIMOS LOS CONJUNTOS PROCESADOS Y LIMPIADOS
datasets <- list(
  "Euskadi" = filtered_datasets[["Euskadi"]],
  "Girona" = filtered_datasets[["Girona"]],
  "Málaga" = filtered_datasets[["Málaga"]],
  "Mallorca" = filtered_datasets[["Mallorca"]],
  "Sevilla" = filtered_datasets[["Sevilla"]],
  "Valencia" = filtered_datasets[["Valencia"]],
  "Menorca" = filtered_datasets[["Menorca"]],
  "Madrid" = filtered_datasets[["Madrid"]],
  "Barcelona" = filtered_datasets[["Barcelona"]]
)
################################################################################
################################################################################
# CAPTURAR CAMPOS CUANTITATIVOS
get_numeric_cols <- function(df, exclude_patterns = c("id", "año", "fecha", "codigo", "code")) {
  numeric_cols <- names(df)[sapply(df, is.numeric)]
  filtered_cols <- numeric_cols[!grepl(paste(exclude_patterns, collapse = "|"), 
                                       tolower(numeric_cols))]
  return(filtered_cols)
}
################################################################################
################################################################################
# FUNCIÓN PARA ANÁLISIS ESTADÍSTICO Y DETECCIÓN DE VALORES ATÍPICOS
analyze_outliers <- function(df, region_name, export_csv = TRUE, plot_results = TRUE) {
  cat("\n=== Análisis de valores atípicos para", region_name, "===\n")
  
  # Identificar campos cuantitativos relevantes
  numeric_cols <- get_numeric_cols(df)
  if (length(numeric_cols) == 0) {
    cat("No se encontraron campos cuantitativos relevantes.\n")
    return(NULL)
  }
  
  # Crear un dataframe para almacenar resultados
  results <- data.frame(
    Variable = character(),
    N = numeric(),
    Media = numeric(),
    Mediana = numeric(),
    Desv_Estandar = numeric(),
    Min = numeric(),
    Max = numeric(),
    Q1 = numeric(),
    Q3 = numeric(),
    IQR = numeric(),
    Limite_Inferior = numeric(),
    Limite_Superior = numeric(),
    N_Outliers = numeric(),
    Pct_Outliers = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Datos para gráficos combinados
  all_outliers_data <- data.frame()
  
  # Análisis por variable
  for (col in numeric_cols) {
    values <- df[[col]]
    valid_values <- values[!is.na(values)]
    n_valid <- length(valid_values)
    
    if (n_valid > 0) {
      # Calcular estadísticas básicas
      mean_val <- mean(valid_values)
      median_val <- median(valid_values)
      sd_val <- sd(valid_values)
      min_val <- min(valid_values)
      max_val <- max(valid_values)
      q1 <- quantile(valid_values, 0.25)
      q3 <- quantile(valid_values, 0.75)
      iqr <- q3 - q1
      
      # Calcular límites para outliers
      lower_bound <- q1 - 1.5 * iqr
      upper_bound <- q3 + 1.5 * iqr
      
      # Identificar outliers
      outliers <- valid_values[valid_values < lower_bound | valid_values > upper_bound]
      n_outliers <- length(outliers)
      pct_outliers <- (n_outliers / n_valid) * 100
      
      # Almacenar resultados
      new_row <- data.frame(
        Variable = col,
        N = n_valid,
        Media = mean_val,
        Mediana = median_val,
        Desv_Estandar = sd_val,
        Min = min_val,
        Max = max_val,
        Q1 = q1,
        Q3 = q3,
        IQR = iqr,
        Limite_Inferior = lower_bound,
        Limite_Superior = upper_bound,
        N_Outliers = n_outliers,
        Pct_Outliers = round(pct_outliers, 2),
        stringsAsFactors = FALSE
      )
      results <- rbind(results, new_row)
      
      # Almacenar datos para gráfico combinado
      if (n_outliers > 0) {
        outlier_df <- df[df[[col]] < lower_bound | df[[col]] > upper_bound, ]
        outlier_df$Variable <- col
        outlier_df$Valor <- outlier_df[[col]]
        all_outliers_data <- rbind(all_outliers_data, 
                                   outlier_df[, c("Variable", "Valor")])
      }
      
      # Imprimir resultados textuales concisos
      cat("\nVariable:", col, "\n")
      cat("Estadísticas: N =", n_valid, "| Media =", round(mean_val, 2), 
          "| Mediana =", round(median_val, 2), "| DE =", round(sd_val, 2), "\n")
      cat("Valores atípicos:", n_outliers, "(", round(pct_outliers, 2), "%)\n")
      
      if (plot_results) {
        # Visualizar distribución y outliers en un solo gráfico
        p <- ggplot(df, aes(x = .data[[col]])) +
          geom_histogram(aes(y = ..density..), binwidth = (max_val - min_val)/30, 
                         fill = "lightblue", color = "darkblue", alpha = 0.7) +
          geom_density(color = "red", linewidth = 1) +
          geom_vline(xintercept = lower_bound, color = "orange", linetype = "dashed", linewidth = 0.8) +
          geom_vline(xintercept = upper_bound, color = "orange", linetype = "dashed", linewidth = 0.8) +
          geom_vline(xintercept = mean_val, color = "blue", linetype = "solid", linewidth = 0.8) +
          geom_vline(xintercept = median_val, color = "darkgreen", linetype = "solid", linewidth = 0.8) +
          labs(title = paste0(region_name, ": Distribución de ", col),
               subtitle = paste0("Outliers: ", n_outliers, " (", round(pct_outliers, 2), "%)"),
               x = col, y = "Densidad") +
          theme_minimal() +
          theme(plot.title = element_text(face = "bold"),
                plot.subtitle = element_text(color = "darkred"))
        
        # Boxplot
        b <- ggplot(df, aes(y = .data[[col]])) +
          geom_boxplot(fill = "lightblue", outlier.color = "red", outlier.size = 2) +
          coord_flip() +
          labs(title = "Boxplot", x = "") +
          theme_minimal()
        
        # Combinar gráficos
        combined_plot <- grid.arrange(p, b, ncol = 1, heights = c(3, 1))
        
        # Guardar el gráfico combinado
        ggsave(filename = paste0("plot_", tolower(region_name), "_", col, "_dist_box.png"),
               plot = combined_plot, width = 8, height = 6, dpi = 300)
      }
    }
  }
  
  # Ordenar resultados por porcentaje de outliers (descendente)
  results <- results[order(-results$Pct_Outliers), ]
  
  # Crear visualización comparativa de todas las variables
  if (nrow(results) > 0 && plot_results) {
    comparison_plot <- ggplot(results, aes(x = reorder(Variable, Pct_Outliers), y = Pct_Outliers)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      geom_text(aes(label = paste0(Pct_Outliers, "%")), vjust = -0.5, size = 3) +
      labs(title = paste("Comparativa de valores atípicos -", region_name),
           subtitle = "Porcentaje de valores atípicos por variable",
           x = "Variable", y = "Porcentaje (%)") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    print(comparison_plot)
    
    # Guardar el gráfico comparativo
    ggsave(filename = paste0("plot_", tolower(region_name), "_comparison.png"),
           plot = comparison_plot, width = 8, height = 6, dpi = 300)
  }
  
  # Exportar resultados
  if (export_csv && nrow(results) > 0) {
    csv_filename <- paste0("outliers_", gsub(" ", "_", tolower(region_name)), ".csv")
    write.csv(results, csv_filename, row.names = FALSE)
    cat("\nResultados exportados a:", csv_filename, "\n")
  }
  
  return(results)
}

# Función para análisis comparativo entre regiones
compare_regions_outliers <- function(all_results) {
  comparison_data <- data.frame()
  
  for (region_name in names(all_results)) {
    region_results <- all_results[[region_name]]
    if (!is.null(region_results) && nrow(region_results) > 0) {
      region_data <- region_results %>%
        select(Variable, Pct_Outliers) %>%
        mutate(Region = region_name)
      comparison_data <- rbind(comparison_data, region_data)
    }
  }
  
  if (nrow(comparison_data) > 0) {
    cat("\n=== Comparativa entre regiones ===\n")
    
    # Gráfico de calor
    heatmap_plot <- ggplot(comparison_data, aes(x = Region, y = Variable, fill = Pct_Outliers)) +
      geom_tile() +
      scale_fill_gradient(low = "white", high = "red", name = "% Outliers") +
      labs(title = "Mapa de calor de valores atípicos por región y variable",
           x = "Región", y = "Variable") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    print(heatmap_plot)
    
    # Guardar el gráfico de calor
    ggsave(filename = "plot_heatmap_regions.png", plot = heatmap_plot, 
           width = 10, height = 6, dpi = 300)
    
    # Análisis por variable
    for (var in unique(comparison_data$Variable)) {
      var_data <- comparison_data[comparison_data$Variable == var, ]
      
      var_plot <- ggplot(var_data, aes(x = reorder(Region, Pct_Outliers), y = Pct_Outliers)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        geom_text(aes(label = round(Pct_Outliers, 1)), vjust = -0.5, size = 3) +
        labs(title = paste("Comparativa de", var, "por región"),
             x = "Región", y = "Porcentaje de outliers (%)") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      print(var_plot)
      
      # Guardar el gráfico por variable
      ggsave(filename = paste0("plot_", tolower(var), "_across_regions.png"),
             plot = var_plot, width = 8, height = 6, dpi = 300)
    }
  }
}

# Función para recomendar tratamiento de outliers
recommend_outlier_treatment <- function(results, threshold_pct = 5) {
  cat("\n=== Recomendaciones para tratamiento de outliers ===\n")
  
  low_outliers <- results[results$Pct_Outliers < threshold_pct, ]
  high_outliers <- results[results$Pct_Outliers >= threshold_pct, ]
  
  if (nrow(low_outliers) > 0) {
    cat("\nVariables con pocos outliers (<", threshold_pct, "%):\n")
    for (i in 1:nrow(low_outliers)) {
      cat("- ", low_outliers$Variable[i], "(", low_outliers$Pct_Outliers[i], "%):",
          "Considerar winsorización o eliminación selectiva.\n")
    }
  }
  
  if (nrow(high_outliers) > 0) {
    cat("\nVariables con muchos outliers (≥", threshold_pct, "%):\n")
    for (i in 1:nrow(high_outliers)) {
      cat("- ", high_outliers$Variable[i], "(", high_outliers$Pct_Outliers[i], "%):",
          "Revisar la distribución. Posible transformación logarítmica o similar.\n")
    }
  }
}

# Función principal de análisis
run_outlier_analysis <- function(datasets, export_csv = TRUE, plot_results = TRUE) {
  all_results <- list()
  for (name in names(datasets)) {
    df <- datasets[[name]]
    result <- analyze_outliers(df, name, export_csv, plot_results)
    if (!is.null(result)) {
      all_results[[name]] <- result
    }
  }
  
  compare_regions_outliers(all_results)
  
  cat("\n=== Resumen de Resultados por Región ===\n")
  for (name in names(all_results)) {
    cat("\nRegión:", name, "\n")
    region_results <- all_results[[name]]
    region_results_with_outliers <- region_results[region_results$N_Outliers > 0, ]
    if (nrow(region_results_with_outliers) > 0) {
      print(region_results_with_outliers[, c("Variable", "N", "N_Outliers", "Pct_Outliers")])
      recommend_outlier_treatment(region_results_with_outliers)
    } else {
      cat("No se encontraron outliers significativos.\n")
    }
  }
  
  return(all_results)
}
################################################################################
################################################################################
outlier_results <- run_outlier_analysis(datasets)














# ANÁLISIS Y TRATAMIENTO DE VALORES ATÍPICOS CON TÉCNICAS AVANZADAS
# LIBRERÍAS NECESARIAS
library(dplyr)      # Manipulación de datos
library(ggplot2)    # Visualización
library(gridExtra)  # Organización de gráficos
library(scales)     # Escalado en gráficos
library(tidyr)      # Transformación de datos
library(MASS)       # Para MAD y Box-Cox

# DATOS DE EJEMPLO (SUSTITUYE CON TUS PROPIOS DATOS)
# Supongamos que tienes una lista de datasets llamada 'filtered_datasets'
# Ejemplo:
# filtered_datasets <- list("Euskadi" = data.frame(a = rnorm(100), b = rexp(100)), ...)
datasets <- list(
  "Euskadi" = filtered_datasets[["Euskadi"]],
  "Girona" = filtered_datasets[["Girona"]],
  "Málaga" = filtered_datasets[["Málaga"]],
  "Mallorca" = filtered_datasets[["Mallorca"]],
  "Sevilla" = filtered_datasets[["Sevilla"]],
  "Valencia" = filtered_datasets[["Valencia"]],
  "Menorca" = filtered_datasets[["Menorca"]],
  "Madrid" = filtered_datasets[["Madrid"]],
  "Barcelona" = filtered_datasets[["Barcelona"]]
)

# FUNCIÓN PARA IDENTIFICAR COLUMNAS NUMÉRICAS
get_numeric_cols <- function(df, exclude_patterns = c("id", "año", "fecha", "codigo", "code")) {
  numeric_cols <- names(df)[sapply(df, is.numeric)]
  filtered_cols <- numeric_cols[!grepl(paste(exclude_patterns, collapse = "|"), tolower(numeric_cols))]
  return(filtered_cols)
}

# FUNCIÓN PARA ANÁLISIS DE OUTLIERS
analyze_outliers <- function(df, region_name, export_csv = TRUE, plot_results = TRUE) {
  cat("\n=== Análisis de valores atípicos para", region_name, "===\n")
  
  numeric_cols <- get_numeric_cols(df)
  if (length(numeric_cols) == 0) {
    cat("No se encontraron campos cuantitativos relevantes.\n")
    return(NULL)
  }
  
  results <- data.frame(
    Variable = character(),
    N = numeric(),
    Media = numeric(),
    Mediana = numeric(),
    Desv_Estandar = numeric(),
    Min = numeric(),
    Max = numeric(),
    Q1 = numeric(),
    Q3 = numeric(),
    IQR = numeric(),
    Limite_Inferior = numeric(),
    Limite_Superior = numeric(),
    N_Outliers = numeric(),
    Pct_Outliers = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (col in numeric_cols) {
    values <- df[[col]]
    valid_values <- values[!is.na(values)]
    n_valid <- length(valid_values)
    
    if (n_valid > 0) {
      # Estadísticas básicas
      mean_val <- mean(valid_values)
      median_val <- median(valid_values)
      sd_val <- sd(valid_values)
      min_val <- min(valid_values)
      max_val <- max(valid_values)
      q1 <- quantile(valid_values, 0.25)
      q3 <- quantile(valid_values, 0.75)
      iqr <- q3 - q1
      
      # Detección de outliers con MAD
      mad_val <- mad(valid_values, constant = 1.4826)
      lower_bound <- median_val - 3 * mad_val
      upper_bound <- median_val + 3 * mad_val
      outliers <- valid_values[valid_values < lower_bound | valid_values > upper_bound]
      n_outliers <- length(outliers)
      pct_outliers <- (n_outliers / n_valid) * 100
      
      # Almacenar resultados
      new_row <- data.frame(
        Variable = col,
        N = n_valid,
        Media = mean_val,
        Mediana = median_val,
        Desv_Estandar = sd_val,
        Min = min_val,
        Max = max_val,
        Q1 = q1,
        Q3 = q3,
        IQR = iqr,
        Limite_Inferior = lower_bound,
        Limite_Superior = upper_bound,
        N_Outliers = n_outliers,
        Pct_Outliers = round(pct_outliers, 2),
        stringsAsFactors = FALSE
      )
      results <- rbind(results, new_row)
      
      # Imprimir resultados
      cat("\nVariable:", col, "\n")
      cat("Estadísticas: N =", n_valid, "| Media =", round(mean_val, 2), 
          "| Mediana =", round(median_val, 2), "| DE =", round(sd_val, 2), "\n")
      cat("Outliers:", n_outliers, "(", round(pct_outliers, 2), "%)\n")
      
      # Visualización
      if (plot_results) {
        p <- ggplot(df, aes(x = .data[[col]])) +
          geom_histogram(aes(y = ..density..), binwidth = (max_val - min_val)/30, 
                         fill = "lightblue", color = "darkblue", alpha = 0.7) +
          geom_density(color = "red", linewidth = 1) +
          geom_vline(xintercept = lower_bound, color = "orange", linetype = "dashed") +
          geom_vline(xintercept = upper_bound, color = "orange", linetype = "dashed") +
          labs(title = paste0(region_name, ": ", col), 
               subtitle = paste0("Outliers: ", n_outliers, " (", round(pct_outliers, 2), "%)"),
               x = col, y = "Densidad") +
          theme_minimal()
        ggsave(paste0("plot_", tolower(region_name), "_", col, ".png"), p, width = 8, height = 6)
      }
    }
  }
  
  # Ordenar por porcentaje de outliers
  results <- results[order(-results$Pct_Outliers), ]
  
  if (export_csv && nrow(results) > 0) {
    csv_filename <- paste0("outliers_", gsub(" ", "_", tolower(region_name)), ".csv")
    write.csv(results, csv_filename, row.names = FALSE)
    cat("Resultados exportados a:", csv_filename, "\n")
  }
  
  return(results)
}

# FUNCIÓN PARA TRATAR OUTLIERS
treat_outliers <- function(df, results, threshold_pct = 5) {
  df_treated <- df
  for (i in 1:nrow(results)) {
    var <- results$Variable[i]
    pct <- results$Pct_Outliers[i]
    
    if (pct < threshold_pct) {
      # Menor al 5%: Winsorización (percentiles 5% y 95%)
      lower_limit <- quantile(df_treated[[var]], 0.05, na.rm = TRUE)
      upper_limit <- quantile(df_treated[[var]], 0.95, na.rm = TRUE)
      df_treated[[var]] <- pmin(pmax(df_treated[[var]], lower_limit), upper_limit)
      cat("Variable", var, "winsorizada (< 5% outliers).\n")
    } else {
      # Mayor o igual al 5%: Transformación logarítmica o Box-Cox
      if (min(df_treated[[var]], na.rm = TRUE) > 0) {
        df_treated[[var]] <- log1p(df_treated[[var]])  # log(1+x) para evitar log(0)
        cat("Variable", var, "transformada logarítmicamente (≥ 5% outliers).\n")
      } else {
        # Si hay valores no positivos, usar Box-Cox con ajuste
        bc <- boxcox(df_treated[[var]] + 1 ~ 1, lambda = seq(-2, 2, 0.1), plotit = FALSE)
        lambda_opt <- bc$x[which.max(bc$y)]
        df_treated[[var]] <- ( (df_treated[[var]] + 1)^lambda_opt - 1 ) / lambda_opt
        cat("Variable", var, "transformada con Box-Cox (≥ 5% outliers).\n")
      }
    }
  }
  return(df_treated)
}

# FUNCIÓN PARA RECOMENDAR TRATAMIENTO
recommend_outlier_treatment <- function(results, threshold_pct = 5) {
  cat("\n=== Recomendaciones para tratamiento de outliers ===\n")
  for (i in 1:nrow(results)) {
    var <- results$Variable[i]
    pct <- results$Pct_Outliers[i]
    cat("\nVariable:", var, "(", pct, "%)\n")
    if (pct < threshold_pct) {
      cat("Recomendación: Winsorización (< 5%).\n")
    } else {
      cat("Recomendación: Transformación logarítmica o Box-Cox (≥ 5%).\n")
    }
  }
}

# FUNCIÓN PRINCIPAL
run_outlier_analysis <- function(datasets, export_csv = TRUE, plot_results = TRUE) {
  all_results <- list()
  treated_datasets <- list()
  
  for (name in names(datasets)) {
    df <- datasets[[name]]
    result <- analyze_outliers(df, name, export_csv, plot_results)
    if (!is.null(result)) {
      all_results[[name]] <- result
      recommend_outlier_treatment(result)
      treated_datasets[[name]] <- treat_outliers(df, result)
    }
  }
  
  cat("\n=== Resumen de Resultados por Región ===\n")
  for (name in names(all_results)) {
    cat("\nRegión:", name, "\n")
    region_results <- all_results[[name]]
    print(region_results[, c("Variable", "N", "N_Outliers", "Pct_Outliers")])
  }
  
  return(list(results = all_results, treated_datasets = treated_datasets))
}

# EJECUTAR ANÁLISIS
outlier_analysis <- run_outlier_analysis(datasets)