####dplyr es parte de tidyverse
install.packages("DT")


library(tidyverse)


library(readxl)
library(dplyr)
library(lubridate)
library(plotly)
library(tidyverse)
library(DT)


my.path <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(my.path)


# https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset


datos1 <- read.csv("https://raw.githubusercontent.com/rday-colombia/tutoriales_Rday_2023/main/07%20Procesamiento%20de%20datos%20con%20dplyr/shopping_trends.csv")


# select: devuelve un conjunto de columnas
# filter: devuelve un conjunto de filas según una o varias condiciones lógicas
# arrange: reordena filas de un data frame
# rename: renombra variables en una data frame
# mutate: añade nuevas variables/columnas o transforma variables existentes
# summarise/summarize: genera resúmenes estadísticos de diferentes 
# variables en el data frame, posiblemente con strata

names(datos1)

# Dejamos variables de interés (select)

datos2 <- datos1 %>%  # pipe (tubería) se incluye ctrl + shift + M
  select(Customer.ID, Age, Gender, Category, Purchase.Amount..USD., Size)


# Cambiar nombres a variables (rename)

datos3 <- datos2 %>% 
  rename(id = Customer.ID, edad = Age, género = Gender,
         valor_compra = Purchase.Amount..USD., talla = Size,
         categoría = Category)


# Dejar personas con edades entre 20 y 60 años 
# y quitar información de predas accesorias (filter)

datos4 <- datos3 %>% 
  filter(edad >= 20,
         edad <= 60,
         categoría != "Outerwear")


# Cambiamos los nombres de las categorías de algunas variables

datos5 <- datos4 %>% 
  mutate(género = if_else(condition = género == "Male",
                          true = "Hombre",
                          false = "Mujer")) %>% 
  mutate(categoría = if_else(condition = categoría == "Clothing",
                             true = "ropa",
                             false = categoría)) %>% 
  mutate(categoría = if_else(condition = categoría == "Accessories",
                             true = "accesorios",
                             false = categoría)) %>% 
  mutate(categoría = if_else(condition = categoría == "Footwear",
                             true = "calzado",
                             false = categoría)) %>% 
  
  # Creamos una nueva variable (mutate), categoría de género y talla
  
  mutate(compuesto = paste0(género, talla))


# Sorteamos los datos por categoría, edad y de manera inversa por 
# valor de compra (arrange, desc)


datos6 <- datos5 %>% 
  arrange(categoría, edad, desc(valor_compra))


# Resumen de variables (summarise)


# Edad


resumen_edad <- datos6 %>% 
  summarise(media = as.numeric(round(mean(edad), 2)),
            desvío = as.numeric(round(sd(edad), 2)),
            mínimo = as.numeric(round(min(edad), 2)),
            máximo = as.numeric(round(max(edad), 2)),
            cv = as.numeric(round((desvío / media) * 100), 2))


# Valor compra


resumen_valor_compra <- datos6 %>% 
  summarise(media = as.numeric(round(mean(valor_compra), 2)),
            desvío = as.numeric(round(sd(valor_compra), 2)),
            mínimo = as.numeric(round(min(valor_compra), 2)),
            máximo = as.numeric(round(max(valor_compra), 2)),
            cv = as.numeric(round((desvío / media) * 100), 2))



# Resúmenes incluyendo dos o más variables por valor compra 
# (summarise, group_by)


# Categoría y género

resumen_cat_gen <- datos6 %>% 
  group_by(categoría, género) %>% 
  summarise(media = as.numeric(round(mean(valor_compra), 2)),
            desvío = as.numeric(round(sd(valor_compra), 2)),
            mínimo = as.numeric(round(min(valor_compra), 2)),
            máximo = as.numeric(round(max(valor_compra), 2)),
            cv = as.numeric(round((desvío / media) * 100), 2),
            n = length(valor_compra))

DT::datatable(resumen_cat_gen,
              filter = 'top')


# Categoría y compuesto

resumen_cat_comp <- datos6 %>% 
  group_by(categoría, compuesto) %>% 
  summarise(media = as.numeric(round(mean(valor_compra), 2)),
            desvío = as.numeric(round(sd(valor_compra), 2)),
            mínimo = as.numeric(round(min(valor_compra), 2)),
            máximo = as.numeric(round(max(valor_compra), 2)),
            cv = as.numeric(round((desvío / media) * 100), 2),
            n = length(valor_compra))

DT::datatable(resumen_cat_comp,
              filter = 'top')


# Edad, categoría y género

resumen_ed_cat_gen <- datos6 %>% 
  group_by(edad, categoría, género) %>% 
  summarise(media = as.numeric(round(mean(valor_compra), 2)),
            desvío = as.numeric(round(sd(valor_compra), 2)),
            mínimo = as.numeric(round(min(valor_compra), 2)),
            máximo = as.numeric(round(max(valor_compra), 2)),
            cv = as.numeric(round((desvío / media) * 100), 2),
            n = length(valor_compra))


DT::datatable(resumen_ed_cat_gen,
              filter = 'top')


# Gráficos


# Edad vs valor compra por género

datos6a <- datos6 %>% 
  mutate(edad = as.factor(edad)) %>% 
  group_by(edad, género) %>% 
  summarise(media = as.numeric(round(mean(valor_compra), 2))) %>% 
  mutate(edad = as.character(edad),
         edad = as.numeric(edad))



g1 <- ggplot(
  datos6a, aes(edad, media, color = género)) +
  geom_point(color = "#66ffc2")+
  geom_line() +
  scale_x_continuous(breaks = seq(
    from = 20,
    to = 60,
    by = 2)) +
  facet_wrap(~ género) +
  labs(x = "Edad (años)", y = "Valor compra (dólares)") +
  theme_minimal() +
  theme(
    plot.title = element_text(color = "black", size = rel(1), face = "bold", hjust = 0.5),
    axis.title.x = element_text(color = "black", size = rel(0.8), face = "bold"), #size=0.8
    axis.title.y = element_text(color = "black", size = rel(0.8), face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size = rel(0.8)),
    axis.text.y = element_text(angle = 0, hjust = 1, size = rel(0.8))) +
  theme_bw()

g1 <- ggplotly(g1)
g1 


# Edad vs valor compra por talla

datos6b <- datos6 %>% 
  mutate(edad = as.factor(edad)) %>% 
  group_by(edad, talla) %>% 
  summarise(media = as.numeric(round(mean(valor_compra), 2))) %>% 
  mutate(edad = as.character(edad),
         edad = as.numeric(edad))



g2 <- ggplot(
  datos6b, aes(edad, media, color = talla)) +
  geom_point(color = "#66ffc2")+
  geom_line() +
  scale_x_continuous(breaks = seq(
    from = 20,
    to = 60,
    by = 2)) +
  facet_wrap(~ talla) +
  labs(x = "Edad (años)", y = "Valor compra (dólares") +
  theme_minimal() +
  theme(
    plot.title = element_text(color = "black", size = rel(1), face = "bold", hjust = 0.5),
    axis.title.x = element_text(color = "black", size = rel(0.8), face = "bold"), #size=0.8
    axis.title.y = element_text(color = "black", size = rel(0.8), face = "bold"),
    axis.text.x = element_text(angle = 90, hjust = 1, size = rel(0.8)),
    axis.text.y = element_text(angle = 0, hjust = 1, size = rel(0.8))) +
  theme_bw()

g2 <- ggplotly(g2)
g2