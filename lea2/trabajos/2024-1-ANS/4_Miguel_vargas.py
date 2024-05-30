# -*- coding: utf-8 -*-
"""Parcial_Analitica_V1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13OeZNM-AYkLLjalXvFj3WTE7VBYRlsWZ

#Parcial Analitica, aprendizaje no supervisado

Autores:

- Simon Marin Zuluaga
- Santiago Cordoba Castañeda
- Miguel Vargas Otalvaro

Instalar Librerias
"""

pip install kneed

pip install factor_analyzer

"""Cargar liberias"""

import seaborn as sns ### para los datos y para gráficar
import matplotlib.pyplot as plt
from sklearn import cluster ### modelos de clúster
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score ## indicador para consistencia de clúster

"""Cargar DF

Diccionario de datos


CUST_ID: Identificación del titular de la tarjeta de crédito

BALANCE: Saldo disponible para compras

BALANCE_FREQUENCY: Frecuencia de actualización del saldo (donde 1 = actualización frecuente y 0 = no se actualiza con frecuencia).

PURCHASES: Importe de las compras realizadas por el cliente

ONEOFF_PURCHASES: Importe máximo de la compra realizada a una couta

INSTALLMENTS_PURCHASES: Importe de las compras realizadas a plazos

CASH_ADVANCE: Avances de Efectivo realizados por el usuario

PURCHASES_FREQUENCY: Frecuencia con la que se realizan las compras (1 = compras frecuentes, 0 = compras poco frecuentes)

ONEOFF_PURCHASES_FREQUENCY: frecuencia con la que se realizan las compras a una couta (1 = compras frecuentes, 0 = compras poco frecuentes)

PURCHASES_INSTALLMENTS_FREQUENCY: Frecuencia con la que se realizan compras a plazos (1 = se realizan con frecuencia, 0 = no se realizan con frecuencia)

CASH_ADVANCE_FREQUENCY: Frecuencia con la que se realizan avances de efectivo

CASH_ADVANCE_TRX: Número de transacciones realizadas por "avances de efectivo"

PURCHASES_TRX: Número de transacciones de compra realizadas

CREDIT_LIMIT: Límite de la tarjeta de crédito del usuario

PAYMENTS: Importe de los pagos realizados por el usuario

MINIMUM_PAYMENTS: Importe mínimo de los pagos realizados por el usuario

PRC_FULL_PAYMENT: Porcentaje del pago total abonado por el usuario

TENURE: Tenencia del servicio de tarjeta de crédito para el usuario
"""

df = pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/credit_card_clients.csv',index_col = 0)

#Corroborar numero de observaciones y variables
df.shape

#Imprimir top registros
print(df.head())

#Conocer tipologia de las variables y datos nullos
print(df.info())

#Conocer estadisticas de las variables
print(df.describe())

#Validar datos nulos
df.isnull().sum()

#Reemplazar datos nulos con cero para la variable MINIMUM_PAYMENTS
df['MINIMUM_PAYMENTS'].fillna(value= 0, inplace=True)

##Eliminar Datos Faltantes, dado el numero de observaciones para CREDIT_LIMIT
# 1 observacion en null
df = df.dropna(subset=['CREDIT_LIMIT'])

#Comprobar cambios
df.isnull().sum()

"""Tomar Variables numericas"""

# Tomar variables númericas
df_num = df.select_dtypes(include = ['int64','float64'])

#Conocer estadisticas de las variables númericas
df_num.describe()

#Conocer distribucción de los datos
ax = df_num.iloc[:,].plot(kind = 'box',title = 'Boxplot variables',showmeans = True,figsize=(10,5))

#Tomar una copia del df original
df_filtro = df_num

#importa libreria
from sklearn.preprocessing import StandardScaler

#Escalar Variables- Codigo se perdio en la definición
df_scaled=StandardScaler().fit_transform(df_num)

df_scaled=pd.DataFrame(df_scaled)

ax = df_scaled.iloc[:,].plot(kind = 'box',title = 'Boxplot variables',showmeans = True,figsize=(10,5))

"""Validar correlación de las variables"""

# Calcula la matriz de correlación.
matriz_correlacion = df_scaled.corr()

# Visualizar la matriz de correlación
plt.figure(figsize=(15, 12), dpi=80)
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", linewidths=0.5, linecolor='white')
plt.title("Mapa de calor")
plt.show()

"""#PCA"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from kneed import KneeLocator

#Esclar variables
sc=StandardScaler ().fit(df_filtro)
feat_sc=sc.transform(df_filtro)

#Aplicación de PCA
pca=PCA(n_components=17) ## se puede dar valor de componentes o varianza explicada
pca.fit(feat_sc)

pca.components_ ## lambdas, vectores propios pesos de observadas sobre latentes
pca.explained_variance_ ## valores propios alpha, cuánta varianza es explicada
ve=pca.explained_variance_ratio_ ### procentaje de variable explicada por cada componente

# Validar peso de las variables originales sobre las latentes
pca.components_

pca.explained_variance_

ve

l = pca.transform(feat_sc) ## variables latentes
l[0] ## variables latentes para primera fila

pd.DataFrame(l)

sns.lineplot(x=np.arange(1,18), y=np.cumsum(ve), palette="viridis")

"""##Punto 1, resultado

Desde el grafico anterior, se eligen dos variables latentes, ya que estas dos variables explican un poco mas 45 % de la variabilidad de los datos. De la tercer variable en adelante es poco significativa en comparativa a las dos primeras.
"""

l_sel=l[:,0:2]

l_sel

sns.scatterplot(x=l_sel[:,0], y=l_sel[:,1])

"""##Punto 2, Resultado

Teniendo en cuenta el contexto del problema, donde nos dan el número de segmentaciones necesarias (3), descartando la aplicación de DBScan por no ver reflejada las diferencias de densidades entre diferentes closters, ademas, no es posible visualizar una sigmentación adecuada de los datos para aplicar GMM, optamos por aplicar un modelo K-Means.
"""

x = range(1, 18)
y = ve[:17]

plt.bar(x, y)
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.title('Primeros dos valores del array')
plt.show()

"""Teniendo el grafico anterior, el porcentaje de varianza que explica los dos primeros componente es del 46%."""

pca.components_

"""Por Analisis de componentes, para la primer variable latente, las tres principales origianles que la explican son:

Primer Variable latente

- Valor de compra de los clientes (40 %)

- Número de transacciones de compras (39 %)

- Valor de las compras, hechas a plazos (34 %)

Segunda Variable Latente

- Avances en Efectivo (43%)
- Frecuencia de los avances (43%)
- Número de transacciones de avances en efectivo (41)
"""



"""####Prueba DBScan"""

from sklearn.neighbors import NearestNeighbors

knn=NearestNeighbors(n_neighbors= 50)
knn.fit(l_sel)
distancia, *_ = knn.kneighbors(l_sel)
distancia_k_mean=np.mean(distancia, axis=0)

k=np.arange(1, 51)

sns.lineplot(x=k, y =distancia_k_mean)

kl=KneeLocator(x=k,y=distancia_k_mean, curve='concave', direction='increasing')
min_sample=kl.elbow ## min sample
eps=distancia_k_mean[kl.elbow] ### eps

print(min_sample)
print(eps)

db=cluster.DBSCAN(eps=eps, min_samples=min_sample)
db.fit(l_sel)

np.unique(db.labels_, return_counts=True)
sns.scatterplot(x=l_sel[:,0],y=l_sel[:,1], hue=db.labels_, palette='viridis')

"""####Prueba GMM"""

from sklearn.model_selection import GridSearchCV
from sklearn import mixture

gmm=mixture.GaussianMixture(n_components=4, covariance_type='full', n_init=5)

gmm.fit(l_sel)

gmm.score(l_sel) ### score por defecto es logaritmo de la función de verosimilitud es adiminesional y no se interpreta por sí solo
gmm.predict_proba(l_sel) ## probabilidad de pertecencer a cada cluster
label=gmm.predict(l_sel) ### para conocer los label del cluster.
gmm.bic(l_sel)

sns.scatterplot(x=l_sel[:,0], y=l_sel[:,1], hue=label, palette='viridis')
plt.title("clusters de acuerdo a gmm")
plt.show()

param_grid={
    'n_components': [2,3],
    'covariance_type':['full', 'diag', 'spherical', 'tied'],
    'n_init': [5, 15, 30]
}

gs=GridSearchCV(gmm, param_grid=param_grid )
gs.fit(l_sel)

gs.cv_results_
gs.best_params_

df_resultados=pd.DataFrame(gs.cv_results_)
pd.set_option('display.max_colwidth',None)
df_resultados.sort_values('mean_test_score', ascending=False)[['params','mean_test_score']]

gmm_win=gs.best_estimator_
label=gmm_win.predict(l_sel)
gmm_win.predict_proba(l_sel)

sns.scatterplot(x=l_sel[:,0], y=l_sel[:,1], hue=label, palette='viridis')
plt.title("clusters de acuerdo a gmm")
plt.show()

"""## Punto 3 - Aplicación K-Means"""

kmeans=cluster.KMeans(n_clusters=3, n_init=10)
kmeans.fit(l_sel)

cluster_label=kmeans.labels_ ### ver los cluster de cada fila
centroides= kmeans.cluster_centers_ ### valores de los centroides

sns.scatterplot(x=l_sel[:,0], y=l_sel[:,1],palette='viridis')

sns.scatterplot(x=l_sel[:,0], y=l_sel[:,1], hue=cluster_label, palette='viridis')

"""El análisis visual de un clustering K-means sobre dos variables latentes revela tres clusters distintos en el conjunto de datos. El primero, situado en la parte inferior izquierda del plano cartesiano, exhibe una alta densidad de puntos cercanos entre sí, indicando una homogeneidad significativa en términos de las variables latentes. Los otros dos clusters, identificados como el superior (morado) y el inferior derecho (amarillo), muestran una densidad de puntos menor y una mayor dispersión, con la presencia de puntos atípicos. El cluster superior e inferior derecho parecen agrupar además los datos atípicos, sugiriendo una menor homogeneidad en comparación con el cluster inferior izquierdo.

## Punto 4
"""

# Agregar etiquetas de clústeres al DataFrame original
df_filtro['Cluster_KMeans'] = cluster_label

# Calcular estadísticas resumidas para cada grupo en función de las características originales
cluster_stats = df_filtro.groupby('Cluster_KMeans').agg(['mean', 'std'])  # Cambiar 'Cluster_KMeans' por 'Cluster_GMM' según corresponda
print(cluster_stats)

# Configurar pandas para mostrar más filas y columnas
import pandas as pd
pd.set_option('display.max_rows', None)  # Mostrar todas las filas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas

# Calcular estadísticas resumidas para cada grupo en función de las características originales
cluster_stats = df_filtro.groupby('Cluster_KMeans').agg(['mean', 'std'])
print(cluster_stats)

"""Con lo evidencia anteriormete.

- Para closter 2, se visualiza un alto importe en las compras con tendencia de compras a una sola cuota, lo cual no genera utilidad en termino de intereses cobrados. Para esto, recomendariamos permitirles a este grupo de usuarios, que una vez fijen la compra a una sola cuota, tener la opción de diferirla a más de 1 couta, hasta antes de la fecha de Corte.

- Para el closter 2, ofrecer un aumento de cupo, ya que son los que tienen menos saldo disponible en la tarjeta, pero son los que realizan mas compras con la misma.

- Para el closter 0, incentivar el uso de la tarjeta de credito, a traves de incentivos por número de compras, como la exoneración o reducción de la cuota de manejo, esto incrementaria la frecuencia de compra, ya que estos se dedican en gran parte a realizar avances.

- Para el closter 1, recomendamos ofrecerse programas de recompensas y descuentos personalizados para fomentar la lealtad y aumentar el gasto

-Para el closter 2, los  clientes son de alto valor que realizan compras con frecuencia y de alto monto. Recomendariamos aplicar estrategias de marketing exclusivas, servicios VIP y experiencias personalizadas para mantener y aumentar la lealtad de estos clientes.

Con lo anterior criterios, podemos definir las disguientes descripciones para los closter.

Closter 0: Cliente Conservador.

Closter 1: Cliente Moderado.

Closter 2: Cliente Premium.


"""

import matplotlib.pyplot as plt

# Suponiendo que cluster_stats es el DataFrame que contiene las estadísticas resumidas
# Calcula la cantidad de características
num_features = len(cluster_stats.columns.levels[0])

# Configura la figura y los ejes
fig, axs = plt.subplots(num_features, 1, figsize=(10, num_features * 5))

# Itera sobre cada característica
for i, feature in enumerate(cluster_stats.columns.levels[0]):
    # Extrae las medias y desviaciones estándar para esta característica
    means = cluster_stats[feature]['mean']
    stds = cluster_stats[feature]['std']

    # Configura la posición de las barras
    x = range(len(means))

    # Grafica las medias con barras y desviaciones estándar como líneas de error
    axs[i].bar(x, means, yerr=stds, capsize=5, align='center', alpha=0.5)
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(means.index)
    axs[i].set_ylabel('Valor')
    axs[i].set_title(f'Estadísticas de "{feature}" por clúster')
    axs[i].grid(True)

plt.tight_layout()
plt.show()