
################################################################################################
################### version 2 #####################################################
###################################################################################

###basicas
import pandas as pd
import numpy as np


##### datos y modelos sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


####### redes neuronales

import tensorflow as tf 
from tensorflow import keras

### cargar datos

url2="https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/iris.csv"
iris_df=pd.read_csv(url2)

### verificar atípicos
iris_df.info()


######## separar x  y y y transofrmar
x_iris=iris_df.iloc[:,0:4]
y_iris=iris_df.iloc[:,4]
x_iris_sc=StandardScaler().fit_transform(x_iris)
x_iris_tr, x_irirs_tes, y_iris_tr, y_irirs_tes=train_test_split(x_iris_sc, y_iris,test_size=0.2, random_state=111)


########## arquitectura de la red #############

ann1=keras.models.Sequential([
    keras.layers.Dense(128,  activation='tanh'),
    keras.layers.Dense(3, activation='softmax')
])


### hiperparmetros de optimización

l=keras.losses.SparseCategoricalCrossentropy()
m=keras.metrics.SparseCategoricalAccuracy()


ann1.compile(loss=l, metrics=m)
ann1.fit(x_iris_tr, y_iris_tr,epochs=10, validation_data=(x_irirs_tes,y_irirs_tes))


###### procedimiento igual que para cualquier modelo


pred_test=np.argmax(ann1.predict(x_irirs_tes), axis=1)

cm=metrics.confusion_matrix(y_irirs_tes, pred_test)
cm_disp=metrics.ConfusionMatrixDisplay(cm)
cm_disp.plot()

print(metrics.classification_report(y_irirs_tes,pred_test))




