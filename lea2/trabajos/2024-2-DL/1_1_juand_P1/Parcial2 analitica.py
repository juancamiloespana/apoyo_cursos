##librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import tensorflow as tf 
from tensorflow import keras
import keras_tuner as kt ### paquete para afinamiento 
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar base de datos
url='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/credit_card_clients.csv'
url1='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/credit_card_clients2.csv'
purchase=pd.read_csv(url)
purchase2=pd.read_csv(url1)

# Verificar nulos
purchase.info()
purchase2.info()
# No hay datos nulos en las BDs
purchase.dropna(inplace=True)
#Separar variable respuesta y explicativa
y= purchase['PURCHASES']
X = purchase.drop(['PURCHASES', 'CUST_ID'], axis = 1) #Se eliminó la variable respuesta y el CUST_ID que era un object

# Cantidad de variables
X.shape
len(y)
y.unique()

# Distribucion de y
plt.hist(y)
y.quantile([0.25, 0.5, 0.75])

##escalado de las variables
pur_sc=StandardScaler().fit(X) ## calcular la media y desviacion para hacer el escalado
X_sc=pur_sc.transform(X)  ## escalado con base en variales escalado
np.isnan(X_sc).sum()


##separar entrenamiento de  evaluación
X_tr, X_te, y_tr, y_te= train_test_split(X_sc, y, test_size=0.2) 
X_tr.shape #Verificar el tamaño de la variable X_tr



##Arquitectura de la red
Ann=keras.models.Sequential([
    keras.layers.InputLayer(shape=X_tr.shape[1:]),  ## capa de entrada 
    keras.layers.Dense(units=1024, activation='sigmoid'), ### capa oculta 1, 1024 neuronas, función de activación sigmoide
    keras.layers.Dense(units=512, activation='relu'), ### capa oculta 2, 512 neuronas, función de activación relu
    keras.layers.Dense(units=256, activation='relu'), ### capa oculta 3, 256 neuronas, función de activación relu
    keras.layers.Dense(units=128, activation='relu'), ### capa oculta 4, 128 neuronas, función de activación relu
    keras.layers.Dense(units=64, activation='relu'),  ### capa oculta 5, 64 neuronas, función de activación relu
    keras.layers.Dense(units=32, activation='relu'), ### capa oculta 6, 32 neuronas, función de activación relu
    keras.layers.Dense(units=16, activation='relu'), ### capa oculta 7, 16 neuronas, función de activación relu
    keras.layers.Dense(units=8, activation='relu'), ### capa oculta 8, 8 neuronas, función de activación relu
    keras.layers.Dense(units=4, activation='relu'), ### capa oculta 9, 4 neuronas, función de activación relu
    keras.layers.Dense(units=1,activation='relu')  # capa de salida
])

# Cantidad de parametros
Ann.count_params()

###compilador
l=keras.losses.MeanSquaredError()
# Crear la métrica MAE
m = keras.metrics.MeanAbsoluteError()

#Optimizador
optimizador_RMS = keras.optimizers.RMSprop(learning_rate=0.1) 

Ann.compile(optimizer= optimizador_RMS, loss=l, metrics=[m])

###configurar el fit
Ann.fit(X_tr, y_tr, epochs=10, validation_data=(X_te, y_te))

Ann.predict(X_tr)
np.isnan(X_tr).sum()

#### hiperparámetros de optimización(entrenamiento)
hp=kt.HyperParameters()


def hyper_mod(hp):
    # Drop out rate
    dor=hp.Float("DOR", min_value=0.01, max_value=0.41, step=0.05)
    # Regularization strenght
    sr=hp.Float("SR", min_value=0.01, max_value=0.021, step=0.002)
    # Opciones de funcion de activacion
    fa=hp.Choice('FA_CO', ['tanh', 'sigmoid', 'relu'])
    # Opciones de optimizador
    opt=hp.Choice('opt', ['Adam', 'SGD', 'RMSprop'])
    # 
    l2_r=keras.regularizers.l2(sr)
    
    
    ##Arquitectura de la red
    Ann=keras.models.Sequential([
    keras.layers.InputLayer(input_shape=X_tr.shape[1:]),  
    keras.layers.Dense(units=1024, activation=fa, kernel_regularizer=l2_r),
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=512, activation=fa, kernel_regularizer=l2_r), 
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=256, activation=fa, kernel_regularizer=l2_r), 
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=128, activation=fa, kernel_regularizer=l2_r), 
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=64, activation=fa, kernel_regularizer=l2_r),  
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=32, activation=fa,kernel_regularizer=l2_r), 
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=16, activation=fa,kernel_regularizer=l2_r),
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=8, activation=fa,kernel_regularizer=l2_r),
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=4, activation=fa, kernel_regularizer=l2_r),
    keras.layers.Dropout(dor),
    keras.layers.Dense(units=1,activation='relu')  
])
    
    l=keras.losses.MeanSquaredError()
    m=keras.metrics.MeanAbsoluteError()
    
    if opt == 'Adam':
        opt1= keras.optimizers.Adam(learning_rate=0.1)
    elif opt == 'SGD':
        opt1= keras.optimizers.SGD(learning_rate=0.1)
    else:
        opt1= keras.optimizers.RMSprop(learning_rate=0.1)
        
    
    Ann.compile(optimizer=opt1, loss=l, metrics=[m])

    return Ann


# Busqueda del modelo con 10 trials
search_model = kt.RandomSearch(
    hypermodel= hyper_mod,
    hyperparameters=hp,
    objective=kt.Objective('val_mean_absolute_error', direction='min'),
    max_trials=10,
    overwrite=True,
    directory="res",
    project_name="afin"
)


### esto es equivalente al fit del modelo 
search_model.search(X_tr, y_tr, epochs=15, validation_data=(X_te, y_te))

search_model.results_summary() ### resultados de afinamiento

model_winner=search_model.get_best_models(1)[0]

model_winner.count_params() ### una red neuronal
model_winner.summary() ### una red neuronal

hps_winner= search_model.get_best_hyperparameters(1)[0]
hps_winner.values

# Predicciones del modelo ganador
y_pred = model_winner.predict(X_te)
y_pred_train = model_winner.predict(X_tr)

y_total=np.array(y) ### para convertir y en array
model=hyper_mod(hps_winner) 
historial = model.fit(X_sc, y_total, epochs=15) ##  queda entrenado el modelo con todos los datos



from sklearn.metrics import mean_absolute_error

# Calcular el Error Medio Absoluto (MAE)
mae1 = mean_absolute_error(y_pred, y_te)
mae2 = mean_absolute_error(y_pred_train, y_tr)

print(f'MAE en el conjunto de entrenamiento: {mae2}')
print(f'MAE en el conjunto de evaluación: {mae1}')

# Calcular el Error
errores = y_pred - np.array(y_te)

# Cuantile

# Guardar objetos
#### se exportan insumos necesarios para prediccion

# Escalador (Se usara para escalar datos nuevos a predecir)
joblib.dump(pur_sc, "escalador_purchases") ### para exporttar objetos de python

model.save('NN_PURCHASES.h5') ## para exportar modelo utilizando la función de tensorflow para evitar conflictos

# Datos de entrenamiento y testeo (Se usara para calcular el desempeño del modelo entrenado)

joblib.dump(X_tr, "X_tr_scaled")
joblib.dump(X_te, "X_te_scaled")
joblib.dump(y_tr, "y_tr")
joblib.dump(y_te, "y_te")