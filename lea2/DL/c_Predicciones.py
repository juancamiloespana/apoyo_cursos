#####paquete básicos ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#### paquetes de sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#### paquetes de redes neuronales

import tensorflow as tf
from tensorflow import keras

### paquete de afinamiento para nn de tensorflor

import keras_tuner as kt

#### paquetes de evaluación de modelos de sklearn

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#####
import joblib

### cargamos los datos 
url='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/iris.csv'

iris_df= pd.read_csv(url)

X_nuevo= iris_df.iloc[:,0:4]



########## cargar la información de escalado y el modelo
win_model_load=joblib.load('C:\\cod\\LEA2\\c_DL\\win_model.joblib')
win_model_load.summary()

sc2=joblib.load("C:\\cod\\LEA2\\c_DL\\sc.joblib")


#####predicciones de datos nuevos ####

X_sc=sc2.transform(X_nuevo)

##

pred_nuevos = win_model_load.predict(X_sc)



