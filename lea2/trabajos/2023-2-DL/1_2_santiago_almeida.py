#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error

#### paquetes de redes neuronales
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt # Paquete para ajustar hiperparametros 

#paquete importacion 
import joblib
# %%
url = 'https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/credit_card_clients.csv'
data = pd.read_csv(url)
data

#TRATAR DATOS 

data.isnull().sum() 
data1 = data.dropna()

data1.isnull().sum()
# %%
y = data1['PURCHASES'] 
"""Deifinmos la variable respuesta ; es decir el monto de compra de cada cliente"""

x = data1.drop(['PURCHASES','CUST_ID'], axis= 1)
"""Seleccionamos las variables explicativas y eliminamos la variable CUST_ID debido a que esta nos puede impedir el escalado del modelo y ademas
puede perjudicar el rendimiento del modelo """
x

# T
# %%

#Escalar variables 

esc = StandardScaler().fit(x)

joblib.dump(esc,"C:/Users/Usuario/Desktop/Analitica/Analitica II/Material/Deep_Learning/esc_trabajoDL.joblib")

X_sc = esc.transform(x)

"""En este apartado exportamos el estandarizado de los datos con el objetivo de que cuando apliquemos el modelo a un nuevo conjunto
de datos no haya necesidad de volver a estandarizar los datos """

#%%
# Separación data

X_tr, X_te, y_tr, y_te  = train_test_split(X_sc, y, test_size= 0.3,random_state= 152) 

X_tr.shape 

"""Separamos los datos 30% para test que equivale a 2685 datos y el porcentaje restante para entrenamiento con el objetivo
de que el modelo tenga una cantidad releevante de datos tanto en train como en test y aumentamos una semilla con random_state
para reducir la variabilidad de las metricas en futuras corridas del codigo"""

# %%
#Primer Diagnostico 

ann1 = keras.models.Sequential([
    keras.layers.Dense(64,input_shape=(16,), activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='relu')
])
"""Definimos una red neuronal con 3 capas con el objetivo de visualizar el rendimiendto del modelo y despues hacer los debidos ajustes
en todas las capa de salida especificamente tuvimos que utilizar la funcion de activacion relu ya que la variable que queremos predecir es una variable numerica continua
la cual no toma valores negativos; el numero de neuronas en la capa de entrada y en las capas oculta fueron elegidos ya que estos numeros son
multiplos de dos y ademas de ser utilizados convencionalmente nos ayudan a optimizar el costo computacional, por otro lado para el input_shape pusimos
el numero de variables explicativas"""


l = keras.losses.MeanSquaredError() 
opt = keras.optimizers.Adam(learning_rate= 0.01) 
"""En este caso utilizamos el MSE como funcion de perdida por el tipo de variable respuesta que tenemos, ademas utilizamos en el 
optimizador Adam ya que es el que ajusta la tasa de aprendaje de forma adaptativa para cada parametro y es el optimizador
que mejor desempeño suele tener"""

m = keras.metrics.MeanAbsoluteError(name = "MAE") 
"""Decidimos utilizar el MAE dado que la variable respuesta contiene ceros por lo cual no es posible usar el MAPE,
ademas el MAE nos permite observar directamente cual es la variación entre los valores reales y los predichos en un valor absoluto"""

y

ann1.compile(optimizer= opt, loss= l, metrics= m) #Definimos los parametros 

ann1.fit(X_tr,y_tr, epochs= 8,validation_data =(X_te, y_te))
"""Elegimos trabajar con 8 epochs para observar el comportamiento de las metricas y la funcion de perdida, en caso de que se estanquen
o si se observa alguna mejora; posteriormente se aumentaran en otro modelo """

ann1.summary()

mape1 = 287/np.mean(y)
mape1

mape2 = 281/np.mean(y)
mape2
"""Dividimos el MAE que obtuvimos en la primera red sobre la media de la variable respuesta para obtener  el MAPE y poder analizar
mejor el rendimiento del modelo, teniendo en cuenta esto el modelo arrojo un MAPE en train de 28% y un test de 
30%, lo cual muestra que el modelo tiene un gran porcentaje de error y aunque en train es mejor; el error sigue siendo alto"""

"""DIAGNOSTICO ann1 : UNDERFITTING"""

#%%

#Diagnostico por grilla  

hp = kt.HyperParameters() #se definen hiperparametros

def mejor_m(hp):
    opti = hp.Choice('OPTI', ['adam','rd2'])
    fa = hp.Choice('FA', ['relu','tanh'])
    
    ann2 = keras.models.Sequential([
    keras.layers.Dense(512, input_shape = (16,), activation=fa),
    
    keras.layers.Dense(256,activation=fa),
    
    keras.layers.Dense(128,activation=fa),
    
    keras.layers.Dense(64,activation=fa),
   
    keras.layers.Dense(32, activation=fa),
 
    keras.layers.Dense(1, activation='relu')
    ])

    if opti == 'adam':
        opti2 = keras.optimizers.Adam(learning_rate=0.001)
    else:
        opti2 = keras.optimizers.RMSprop(learning_rate=0.001) 
    
    ann2.compile(optimizer= opti2, loss= l, metrics= m)
    
    return ann2
"""Con el primer diagnostico decidimos aumentar el numero capas, disminuir la tasa de aprenzaje y mediante una funcion se definio
 los optimizadores y las funciones de activacion van a iterar en la funcion de busqueda del mejor modelo mediante RandomSearch """

#%%
# ANALISIS MODELO GANADOR 

search_model = kt.RandomSearch(
    hypermodel= mejor_m ,
    hyperparameters= hp,
    objective= kt.Objective('val_MAE',direction= 'min'),
    max_trials= 10,
    overwrite = True,
    project_name = 'rest_afin'
)

search_model.search(X_tr,y_tr,epochs = 20,validation_data=(X_te,y_te)) 
search_model.results_summary()
"""Meadiante RandomSearch realizamos 10 iteraciones aleatorias variando los optimizadores y las funciones de activacion, el objetivo de esta busqueda
es encontrar un modelo que minimice el MAE en test; ademas de esto decidimos aumentar el numero de epochs a 20 ya que de esta forma es mas probable que 
el modelo encuentre una mejora"""
# %%
#Selección mejor modelo
win_model = search_model.get_best_models(1)[0]
win_model.build()
win_model.summary()

"""Seleccionamos el modelo con mejor rendimiento de la lista que nos arroja la funcion search.model ya que esta funcion los ordena de forma
descendente de acuerdo a su rendimiento"""

#Exportacion modelo

joblib.dump(win_model,'C:/Users/Usuario/Desktop/Analitica/Analitica II/Material/Deep_Learning/modelo_trabajo.joblib')

"""Exportamos el modelo para que de esta forma al realizar futuros pronosticos  no sea necesario realizar 
todas las corridad de diagnostico y ajuste de nuevo"""
# %%
#Predicciones

predicciones = win_model.predict(X_te)

#Transformación
y_pred = np.array(predicciones)[:,0]
y_test = np.array(y_te)

y_pred.shape
y_test.shape

# Calcular las metricas de desempeño 

metrics.r2_score(y_test,y_pred) #R2

metrics.mean_absolute_error(y_test,y_pred) #MAE

metrics.PredictionErrorDisplay.from_predictions(y_test,y_pred,kind= 'actual_vs_predicted')
"""Tomamos como metricas bases el R2, el cual nos indica que el modelo tiene un desempeño del 88%;
esto quiere decir que el monto de las compras de los usuarios de tarjeta de credito se ven explicados 
en un 88% por el modelo y el 12% restante no."""

"""El MAE nos da muy similar al MAE obtenido en el modelo ganador, esto nos da un indicio de que el modelo
esta correcto"""

"""Realizamos una grafica para mirar el comportamiento de los valores actuales vs los predichos, observando la grafica
nos damos cuenta que se esta teniendo una buena prediccion del monto de compras; sin embargo tambien se observan que
hay ciertos valores atipicos por lo cual esto puede ser un indicio de que el MAE y el MAPE calculados sean alto; por lo cual
habria que hacer un tratamiento sobre los datos atipicos para mirar si con esto se presenta un mejor rendimiento """



# %%
""" PARTE 2 """

#CARGAR NUEVO DATASET

url = 'https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/credit_card_clients2.csv'
data_nueva = pd.read_csv(url)
data_nueva

"""Procedemos a eliminar la variable CUST_ID, ya que esta variable nos puede afectar en las predicciones del modelo porque
el modelo fue entrenado sin CUST_ID; de igual forma esto no afectaria la interpretacion ya que el CUST_ID es una primary key
asi que podemos identificar al cliente usando las posiciones de las tablas"""

data2 = data_nueva.drop(['CUST_ID'],axis = 1)
data2
# %%
#Carga escalado y modelado 

win_model_load = joblib.load('C:/Users/Usuario/Desktop/Analitica/Analitica II/Material/Deep_Learning/modelo_trabajo.joblib')

sc2 = joblib.load('C:/Users/Usuario/Desktop/Analitica/Analitica II/Material/Deep_Learning/esc_trabajoDL.joblib')

"""Importamos el modelo y la escalada de datos definidas anteriormente para hacer las predicciones y tener una referencia de 
escalado para la base de datos nueva"""
#%%
# Escalado de datos 

X_sc2 = sc2.transform(data2)

"""Escalamos los datos en la base de datos nueva"""

#%%
"""Analice los errores del modelo"""
#ANALISIS RESIDUALES

metrics.PredictionErrorDisplay.from_predictions(y_test,y_pred, kind = 'residual_vs_predicted')

resid = y_test-y_pred

np.mean(resid) #Es mas grande el predicho que el actual; el modelo esta SUBESTIMANDO
np.std(resid)

"""Calculando la media de los residuales podemos observar que el modelo esta SUBESTIMANDO
datos, es decir las predicciones de compras del modelo son menores a las compras reales"""
#%%
plt.hist(resid,bins=200)
plt.xlim(-5000,7000)

"""Analizando el histograma, definimos inicialmente 200 intervalos debido a que se tiene un rango
de valores bastante grande, de esta forma tendremos una mejor visibilad de la distribución de los residuales;
una vez dicho esto, en el histograma podemos observar que los residuales tienden a distribuise de forma normal;
concluyendo un buen ajuste de los datos lo cual indica que los errores no están sesgados y que el modelo está capturando
la tendencia central de los datos."""

#%%

"""PREDICCIÓN JUL-DIC 2023"""

pred_nuev = win_model_load.predict(X_sc2)
pred_nuev
"""Cargamos el modelo realizado en la parte 1 para poder realizar las predicciones a la base de datos nueva"""

data3 = pd.DataFrame(pred_nuev).rename(columns= {0:'compras'})
data3
"""Transformamos las predicciones a formado DataFrame con el objetivo de visualizar mejor las predicciones"""

data1

np.sum(data3['compras']) #Suma de compras para compras JUL_DIC 2023
np.sum(data1['PURCHASES']) #Suma de compras para compras JUL-DIC 2022

"""Podemos observar que las compras realizadas en el periodo de JUL_DIC 2022 son mayores a las compras
que se realizaran en el periodo JUL_DIC2023, con esto segun el modelo tenemos una tendencia a que los usuarios
bajen su tasa de compra por lo cual necesitariamos aplicar estrategias para aumentar las compras de los usuarios
en el periodo JUL.DIC del año 2023"""
#%%
""" 5 CLIENTES CON MAYOR SUMA DE COMPRAS E INTERVALO DE CONFIANZA """

top5 = data3.sort_values(by = 'compras', ascending = False).head(5)
top5

de=np.std(resid)
#%%
# INTERVALO 95% de confianza
top5['LI'] = top5['compras'].apply(lambda x: x-3*de) 
top5['LS'] = top5['compras'].apply(lambda x: x+3*de)
top5
"""Aqui calculamos la d.e en base a los residuales, para tener la interpretacion de todo el modelo, usamos la funcion
.apply para que se realice la operacion indicada para cada uno de los clientes(filas)"""

"""En la tabla podemos encontran en la columna LI el limite interior del intervalo de confianza, en la columna
LS el limite superior y la columna compras refleja el valor predicho, por lo cual este valor se encontrara con un
95% de confianza entre LI Y LS """

#INTERPRETACION CLIENTE 1256; esta interpretacion se realiza analogamente para los 5 clientes que mas compras realizaran
"""Con un intervalo de confianza del 95% podemos asegurar que el monto de compras del cliente 1256 en el periodo
JUL-DIC-2023 estara entre 36825.33 y 41586.40"""
# %%

"""Estrategia 1: A los clientes de mayores compras ofrecerles beneficios para fidelizarlos como pueden ser
descuentos de intereses, regalos en sus cumpleaños o darles servicios exclusivos como pueden ser tarjetas
exclusivas con las cuales el cliente pueda tener accesos a diferentes beneficios como : acceso a salas VIP en aereopuertos,
Descuentos en pasajes de avión, accsesos a eventos exclusivos, acceso a la preventa de espectaculos etc..."""

"""Estrategia 2 : Para lograr que los clientes de bajo monto de compras aumenten es importante que aprendan como funciona el sistema financiero 
y el porque usar una tarjeta de crédito para realizar sus comprar NO es algo malo y aclarar los beneficios que les pueden trer, también se pueden dar descuentos en los intereses que pagan
al realizar sus compras cuando estas superan cierto monto de dinero, ademnas podriamos realizar alianzas con diversas tiendas en las cuales si se paga con
tarjeta de crédito se reciben descuentos; ademas de ello tambien se podria aplicar una estrategia de cashback del 5% en cada compra realizada con la tarjeta,
sin embargo al proponer el uso de cashback es necesario consultarlo con el dpto financiero para mirar la viabilidad de la propuesta"""
