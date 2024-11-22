
###basicas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##### datos y modelos sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import sklearn

####### redes neuronales

import tensorflow as tf 
from tensorflow import keras

import keras_tuner as kt
########## diabetes ##########


url="https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/diabetes.csv"
diabetes=pd.read_csv(url)

diabetes.info()

diabetes2=diabetes.drop('progression', axis=1)
diabetes2=diabetes2.sample(n=200)
diabetes2.to_csv('diabetes2.csv', index=False)
diabetes2.info()
diabetes.info()

###separar y y x

y_diab=diabetes['progression']
X_diab=diabetes.iloc[:,0:10]

sc=StandardScaler().fit(X_diab)
X_diab_sc=sc.transform(X_diab)

X_diab_tr, X_diab_te, y_tr, y_te=train_test_split(X_diab_sc, y_diab, test_size=0.2)


dor=0.5
sr = 0.1 ## por defecto
l2=keras.regularizers.l2(sr)

ann2=keras.models.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=l2),
    keras.layers.Dropout(dor),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="relu")   
])

ann2.summary()


m2=keras.metrics.MeanAbsolutePercentageError(name='mape2')

ann2.compile(optimizer="adam", loss=keras.losses.MeanSquaredError(),metrics=[m2])
ann2.fit(X_diab_tr, y_tr, epochs=25, validation_data=(X_diab_te,y_te))



# ##### afinamiento


# hp=kt.HyperParameters()
# lo=keras.losses.MeanSquaredError()
# name_metr="prueba"
# m4=keras.metrics.MeanAbsolutePercentageError(name=name_metr)

# def tun_model(hp):
    
#     dor=hp.Float('DOR', min_value=0.1, max_value=0.6, step=0.1)
#     opt=hp.Choice('opt', ['adam','sgd'])
    
    
#     ann2=keras.models.Sequential([
#         keras.Input(shape=(10,)),
#         keras.layers.Dense(128, activation='relu', kernel_regularizer=l2),
#         keras.layers.Dropout(dor),
#         keras.layers.Dense(64, activation='relu'),
#         keras.layers.Dense(32, activation="relu"),
#         keras.layers.Dense(1, activation="relu")   
#     ])


#     if opt== 'adam':
#         opt=keras.optimizers.Adam(learning_rate=0.001)
#     else:
#         opt=keras.optimizers.SGD(learning_rate=0.001)
        
#     ann2.compile(optimizer=opt, loss=lo, metrics=[m4] )
   
#     return ann2


# search_model=kt.RandomSearch(
#     hypermodel=tun_model,
#     hyperparameters=hp,
#     objective=kt.Objective([name_metr], direction="min"),
#     max_trials=20,
#     overwrite=True,
#     project_name="resultados",
    
# )

# search_model.search(X_diab_tr, y_tr, epochs=10, validation_data=(X_diab_te,y_te))
# search_model.results_summary()

# best_model=search_model.get_best_models(num_models=3)[0]
# hps=search_model.get_best_hyperparameters(5)[0]
# hps.values


# best_model.build()
# best_model.summary()







################### evaluación #####

pred_diab_te=ann2.predict(X_diab_te)
pred_diab_te.shape

y_actual=np.array(y_te)
y_actual.shape

y_pred=np.array(pred_diab_te)[:,0]
y_pred.shape

metrics.PredictionErrorDisplay.from_predictions(y_true=y_actual,y_pred=y_pred, kind="actual_vs_predicted")
metrics.PredictionErrorDisplay.from_predictions(y_true=y_actual,y_pred=y_pred, kind="residual_vs_predicted")


error=y_actual-y_pred



sns.histplot(x=error, bins=30)
metrics.mean_absolute_percentage_error(y_actual,y_pred)
metrics.mean_absolute_error(y_actual,y_pred)


###### intevalo de confianza
sd_inf=np.quantile(error,0.15)
sd_sup=np.quantile(error,0.75)


int_sups=y_pred+sd_sup
int_inf= y_pred+sd_inf


pred_int=pd.DataFrame()
pred_int['inf']=int_inf
pred_int['fit']=y_pred
pred_int['sup']=int_sups

np.mean(error) ##subestimando



######## predicciones a base nueva ######

diabetes2=pd.read_csv("https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/diabetes2.csv")

diabetes
X2sc=sc.transform(diabetes2)
pred_new=ann2.predict(X2sc)
diabetes2["pred"]=pred_new


diabetes2.sort_values('pred', ascending=False)


