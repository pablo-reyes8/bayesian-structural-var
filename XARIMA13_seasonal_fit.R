install.packages("seasonal")
library(seasonal)
library(readr)
library(ggplot2)
library(zoo)
library(gridExtra) 

plot_desestacionalizada <- function(original_ts, ajustada_ts,
                                    tipo = c("base", "ggplot2"),
                                    colores = c("Original" = "black", "Desestacionalizada" = "blue"),
                                    titulo  = "Serie Original vs Desestacionalizada",
                                    xlab    = "Tiempo",
                                    ylab    = "Valor") {
  tipo <- match.arg(tipo)
  
  if (tipo == "base") {
    plot(
      original_ts,
      type = "l",
      col  = colores["Original"],
      lwd  = 2,
      xlab = xlab,
      ylab = ylab,
      main = titulo
    )
    lines(
      ajustada_ts,
      col  = colores["Desestacionalizada"],
      lwd  = 2
    )
    legend(
      "topleft",
      legend = c("Original", "Desestacionalizada"),
      col    = colores,
      lwd    = 2,
      bty    = "n"
    )
    
  } else {
    # ggplot2
    if (!requireNamespace("ggplot2", quietly = TRUE) ||
        !requireNamespace("zoo",      quietly = TRUE)) {
      stop("Necesitas instalar los paquetes ggplot2 y zoo para usar esta opción.")
    }
    
    df <- data.frame(
      fecha    = as.Date(as.yearqtr(time(original_ts))),
      Original = as.numeric(original_ts),
      Ajustada = as.numeric(ajustada_ts)
    )
    
    ggplot(df, aes(x = fecha)) +
      geom_line(aes(y = Original,   color = "Original"),       size = 1) +
      geom_line(aes(y = Ajustada,   color = "Desestacionalizada"), size = 1) +
      scale_color_manual(
        "", 
        values = colores
      ) +
      labs(
        title = titulo,
        x     = xlab,
        y     = ylab
      ) +
      theme_minimal()
  }
}


ajusta_y_grafa_seasonal <- function(data_in, col_name,
                                    start = c(2003,1),
                                    freq  = 4,
                                    tipo  = "ggplot2",
                                    colores = c("Original" = "darkgreen",
                                                "Desestacionalizada" = "orange")) {
  ts_orig <- ts(data_in[[col_name]],
                start     = start,
                frequency = freq)
  
  
  ###########
  aj     <- seas(ts_orig)
  ts_sa  <- final(aj)
  ##############
  
  
  new_col <- paste0(col_name, "_sa")
  data_in[[new_col]] <- as.numeric(ts_sa)
  
  ts_sa_ts <- ts(ts_sa,
                 start     = start,
                 frequency = freq)
  
  title1 <- paste0("Serie original vs Desestacionalizada (", col_name, ")")
  title2 <- paste0("Componente Estacional de ", col_name, " (SEATS)")
  
  p1 <- plot_desestacionalizada(
    original = ts_orig,
    ajustada = ts_sa_ts,
    tipo     = tipo,
    colores  = colores
  ) +
    ggtitle(title1)
  
  est_ts <- series(aj, "seats.seasonal")
  df_est  <- data.frame(
    fecha = as.Date(as.yearqtr(time(est_ts))),
    valor = as.numeric(est_ts)
  )
  p2 <- ggplot(df_est, aes(x = fecha, y = valor)) +
    geom_line(color = "darkorange", size = 1) +
    labs(
      title = title2,
      x     = "Fecha",
      y     = "Magnitud estacional"
    ) +
    theme_minimal()
  
  grid.arrange(p1, p2, ncol = 1)
  
  return(data_in)}


#### How to Use ########### 

Base_Seasonal <- read_csv("Opciones Datos/Base_Seasonal.csv")

intermedio1 <- ajusta_y_grafa_seasonal(Base_Seasonal, "TES")

intermedio2 <- ajusta_y_grafa_seasonal(intermedio1, "Recaudo Estado")

intermedio3 <- ajusta_y_grafa_seasonal(intermedio2, "Inflación total")

intermedio4 <- ajusta_y_grafa_seasonal(intermedio3, "Gasto Total")

intermedio5 <- ajusta_y_grafa_seasonal(intermedio4, "Razón impuestos/PIB")

base_final_sa <- ajusta_y_grafa_seasonal(intermedio5, "Razón gasto publico/PIB")







