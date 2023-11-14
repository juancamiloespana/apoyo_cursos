import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


#### cargar datos mnist para trabajo ###

(x_train, y_train), (x_test, y_test)= keras.datasets.mnist.load_data()

### tensor flow trabaja con arrays no con data frame, si se tiene un datra frame de pandas hay que convertirlo a array
y_train.shape
x_train.shape  
x_test.shape
y_test.shape


#### cargar datos profesor

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train.shape ### los numeros se leen de derecha izquierda si miramos lo que imprime
x_train[1][12][27]
np.unique(y_train, return_counts=True)

#### visualizar  ####
sample=5

plt.imshow(x_train[sample], cmap='gray')
plt.show()

y_train[sample]

####### Noirmalizar datos ######

np.max(x_train)
np.min(x_train)

x_train2=x_train/255
x_test2=x_test/255


### arquitectura de la red neuronal  vs modelo de red neuronal #####
n_o=x_train.shape[0]
filas=x_train.shape[1]
col= x_train.shape[1]
f_x_c = filas*col

outputs=np.unique(y_train).shape[0]

x_train_rs=x_train2.reshape(n_o,f_x_c)
x_train_rs.shape

### crear modelos con su estructura ###

rf = RandomForestClassifier(n_estimators=50)

ann1=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[filas, col]),
    keras.layers.Dense(256, activation='tanh'),
    keras.layers.Dense(outputs, activation= 'softmax')  
])

ann1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics= ['accuracy'])

###### ajustar modelo definido a datos

rf.fit(x_train_rs, y_train)
ann1.fit(x_train2, y_train, epochs=10, validation_data=(x_test2,y_test))


##################################

prediction=ann1.predict(x_test2)
predicted_label=np.argmax(prediction, axis=1)
label=np.unique(predicted_label)

from sklearn import metrics

cm=metrics.confusion_matrix(y_test, predicted_label, labels=label)
disp=metrics.ConfusionMatrixDisplay(cm)
disp.plot()

print(metrics.classification_report(y_test, predicted_label))


y_train2=np.where(y_train>5,1,0)

 
fc_model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[2,2]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

fc_model.count_params()

