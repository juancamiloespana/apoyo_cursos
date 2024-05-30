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

y_pred = win_model.predict(X_te)

# Calcular las metricas de desempeño 

metrics.r2_score(y_te,y_pred) #R2

metrics.mean_absolute_error(y_te,y_pred) #MAE

metrics.PredictionErrorDisplay.from_predictions(y_te,y_pred,kind= 'actual_vs_predicted')

"""Tomamos como metricas bases el R2, el cual nos indica que el modelo tiene un desempeño del 88%;
esto quiere decir que el monto de las compras de los usuarios de tarjeta de credito se ven explicados 
en un 88% por el modelo y el 12% restante no."""

"""El MAE nos da muy similar al MAE obtenido en el modelo ganador, esto nos da un indicio de que el modelo
esta correcto"""

"""Realizamos una grafica para mirar el comportamiento de los valores actuales vs los predichos, observando la grafica
nos damos cuenta que se esta teniendo una buena prediccion del monto de compras; sin embargo tambien se observan que
hay ciertos valores atipicos por lo cual esto puede ser un indicio de que el MAE y el MAPE calculados sean alto; por lo cual
habria que hacer un tratamiento sobre los datos atipicos para mirar si con esto se presenta un mejor rendimiento """