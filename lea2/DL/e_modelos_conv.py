#####librerias
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

##### para afinamiento #########

import keras_tuner as kt

######## cargar datos #####

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train.shape
y_train.shape
np.unique(y_train, return_counts=True)


x_test.shape
y_test.shape


plt.imshow(x_train[7000],cmap='gray')
y_train[7000]
plt.show()

##### escalar variables #####
x_trains=x_train/255
x_tests=x_test/255

f=x_train.shape[1]
c=x_train.shape[2]
fxc= f*c

######### red convolucional #######
fa='tanh' ### fucnion de activación para todas las capas

cnn1=keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation=fa, input_shape=[f,c,1]),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=fa),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation=fa),
    keras.layers.Dense(10, activation="softmax")
    
])


lr=0.002 ## tasa de aprendizaje define si mueve mucho los parámetros o no
optim=keras.optimizers.Adam(lr) ### se configar el optimizador
cnn1.compile(optimizer=optim, loss="sparse_categorical_crossentropy",metrics=['accuracy'] )
cnn1.fit(x_trains, y_train, epochs=20,validation_data=(x_tests, y_test), batch_size=30)


#############################Afinamiento de hiperparámetros

hp=kt.HyperParameters()

def build_model(hp):
    
    ####### definición de hiperparámetros de grilla 
    
    num_conv_layers = hp.Int('num_conv_layers', min_value=1, max_value=3)
    conv_filters = [hp.Int(f'conv_{i+1}_filter', min_value=1, max_value=32, step=16) for i in range(num_conv_layers)]
    conv_kernels = [hp.Choice(f'conv_{i+1}_kernel', values=[3, 1]) for i in range( num_conv_layers)]
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    dense_units = hp.Int('dense_units', min_value=8, max_value=64, step=32) 
    
    ####### creación de modelo sequential vacío y capa de entrada

    model = keras.models.Sequential()### se crea modelo sin ninguna capa
    model.add(keras.layers.InputLayer(input_shape=(f, c, 1))) ### se crea capa de entrada
    
    ##### agregar capas convolucionales de acuerdo a hiperparáemtro de capas
    
    for i in range( num_conv_layers):
        model.add(keras.layers.Conv2D(filters=conv_filters[i], kernel_size=(conv_kernels[i], conv_kernels[i]), activation=activation))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    
    ### agregar capas densas siempre estándar al final de la red 
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=dense_units, activation=activation))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

tuner = kt.RandomSearch(
    build_model,
    hyperparameters=hp,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    overwrite=True,
    project_name='cnn_tuning'
)

tuner.search(x_train, y_train, epochs=1, validation_data=(x_test, y_test), batch_size=600)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model.summary()
best_hps.values


# ######bosques aleatorios ####
# n_obs_tr=x_train.shape[0]  ### redimensionar el array para RF
# n_obs_te =x_test.shape[0]
# f=x_train.shape[1]
# c=x_train.shape[2]
# fxc= f*c

# x_train_rf =x_trains.reshape(n_obs_tr,fxc) ## redimensionar el array para que quede un vector por observacion
# x_test_rf= x_tests.reshape(n_obs_te,fxc)

# x_train_rf.shape  ## verificar que el redimensionamiento sea correcto
# x_test_rf.shape


# ########Ajustar el modelo ####
# rf1=RandomForestClassifier(min_samples_leaf=10) ##crear modelo con hiperparámetros
# rf1.fit(x_train_rf, y_train) ## ajustar a datos o entrenar con datos

# ### analizar  desempeño
# pred_tr= rf1.predict(x_train_rf) ### predicciones en entrenamiento
# pred_te =rf1.predict(x_test_rf) ## predicciones en evaluacion

# print(metrics.classification_report(y_train, pred_tr)) ## metricas en entrenamiento 
# print(metrics.classification_report(y_test, pred_te)) ## metricas en evaluacion

# cm_tr=metrics.confusion_matrix(y_train, pred_tr) ### para crear matriz de confusión entrenamiento
# cm_te = metrics.confusion_matrix(y_test, pred_te) ### para crear matriz de confusión evaluación

# ### para graficar matriz de confusion de entrenamiento
# disp_tr=metrics.ConfusionMatrixDisplay(cm_tr)
# disp_tr.plot()

# ### para graficar matriz de confusion de evaluacion
# disp_tr=metrics.ConfusionMatrixDisplay(cm_te)
# disp_tr.plot()


# ############################################
# ######### red neuronal inicial pequeña ###################################


fa='tanh' ### fucnion de activación para todas las capas

ann1=keras.models.Sequential([
     keras.layers.Flatten(input_shape=[f,c]),
     keras.layers.Dense(128, activation=fa),
     keras.layers.Dense(32, activation=fa),
     keras.layers.Dense(10, activation="softmax")
 ])

ann1.count_params() ### contar parametros del modelo

lr=0.002 ## tasa de aprendizaje define si mueve mucho los parámetros o no
optim=keras.optimizers.Adam(lr) ### se configar el optimizador

ann1.compile(optimizer=optim, loss="sparse_categorical_crossentropy",metrics=['accuracy'] )
ann1.fit(x_trains, y_train, epochs=10,validation_data=(x_tests, y_test), batch_size=30)



# ####Segunda red teniendo en cuenta recomendaciones para underfitting #######


# fa='tanh' ### fucnion de activación para todas las capas

# ann2=keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[f,c]),
#     keras.layers.Dense(256, activation=fa),
#     keras.layers.Dense(128, activation=fa),
#     keras.layers.Dense(64, activation=fa),
#     keras.layers.Dense(32, activation=fa),
#     keras.layers.Dense(10, activation="softmax")
# ])

# ann2.count_params() ### contar parametros del modelo

# lr=0.002 ## tasa de aprendizaje define si mueve mucho los parámetros o no
# optim=keras.optimizers.Adam(lr) ### se configar el optimizador

# ann2.compile(optimizer=optim, loss="sparse_categorical_crossentropy",metrics=['accuracy'] )
# ann2.fit(x_trains, y_train, epochs=20,validation_data=(x_tests, y_test), batch_size=30)


# ### terccera red aplicando regularizacion ###### 

# dor=0.05 ## hiperparámetro de borrado de nuronas, porcentaje de neuronas que va a borrar por capa
# sr=0.001 ## fuerza de la penalizacion valor por defecto 0.01
# l2=keras.regularizers.l2(sr) ### creo el objeto regularizador

# fa='tanh' ### fucnion de activación para todas las capas

# ann2=keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[f,c]),
#     keras.layers.Dense(256, activation=fa, kernel_regularizer=l2 ),
#     keras.layers.Dropout(dor),
#     keras.layers.Dense(128, activation=fa, kernel_regularizer=l2),
#     keras.layers.Dropout(dor),
#     keras.layers.Dense(64, activation=fa),
#     keras.layers.Dense(32, activation=fa),
#     keras.layers.Dense(10, activation="softmax")
# ])

# ann2.count_params() ### contar parametros del modelo

# lr=0.002 ## tasa de aprendizaje define si mueve mucho los parámetros o no
# optim=keras.optimizers.Adam(lr) ### se configar el optimizador

# ann2.compile(optimizer=optim, loss="sparse_categorical_crossentropy",metrics=['accuracy'] )
# ann2.fit(x_trains, y_train, epochs=20,validation_data=(x_tests, y_test), batch_size=30)



# ######### definir grilla de hiperparámetros

# import keras_tuner as kt


# def model_afinar(hp):
#     dr = hp.Float('DOR', min_value=0.02, max_value=0.06, step=0.01)
#     rs=hp.Float('SR', min_value= 0.0001, max_value=0.001, step=0.0002)
#     lr=hp.Float('lr', min_value= 0.001, max_value=0.005, step=0.001)
#     opt= hp.Choice('Opt', ['adam', 'sgd'])
#     l2=keras.regularizers.l2(sr) ### creo el objeto regularizador

#     fa='tanh' ### fucnion de activación para todas las capas

#     ann2=keras.models.Sequential([
#         keras.layers.Flatten(input_shape=[f,c]),
#         keras.layers.Dense(256, activation=fa, kernel_regularizer=l2 ),
#         keras.layers.Dropout(dor),
#         keras.layers.Dense(128, activation=fa, kernel_regularizer=l2),
#         keras.layers.Dropout(dor),
#         keras.layers.Dense(64, activation=fa),
#         keras.layers.Dense(10, activation="softmax")
#     ])
#     if opt=='adam':
#         optim=keras.optimizers.Adam(lr) ### se configar el optimizador
#     else:
#         optim=keras.optimizers.SGD(lr)  
    
#     ann2.compile(optimizer=optim, loss="sparse_categorical_crossentropy",metrics=['accuracy'] )
    
#     return ann2
    

# hp=kt.HyperParameters()  ## decirle que hp es obejto de hiperparámetros de jeras tunner
# model_afinar(hp) ## instanciar el model


# tunner = kt.RandomSearch(
#     hypermodel=model_afinar,
#     hyperparameters= hp,
#     objective=kt.Objective('val_accuracy', direction='max'),
#     max_trials=2    
# )

# tunner.search(x_trains, y_train, epochs=5,validation_data=(x_tests, y_test), batch_size=100)


#tunner.results_summary()
#ann_winner=tunner.get_best_models(num_models=2)[0]

#pred=np.argmax(ann_winner.predict(x_tests), axis=1)

#print(metrics.classification_report(y_test,pred))

