
################################################################################################
################### version 2 #####################################################
###################################################################################

###basicas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##### datos y modelos sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

####### redes neuronales

import tensorflow as tf 
from tensorflow import keras

import keras_tuner as kt



tf.random.set_seed(123)

ann1=keras.models.Sequential([
    keras.layers.Dense(input_shape=(2,),units=2),
    keras.layers.Dense(units=3)
])


ann1.count_params()


##### llevar datos a github

# iris=datasets.load_iris()
# iris.target
# iris_df=pd.DataFrame(data=iris.data, columns=iris.feature_names)

# iris_df['type']=iris.target

# diabetes=datasets.load_diabetes()
# diabetes.target
# diab_df=pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
# diab_df['progression']=diabetes.target

# iris_df.to_csv('iris.csv', index=False)
# diab_df.to_csv('diabetes.csv', index=False)

# import os
# os.getcwd()



url2="https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/iris.csv"
iris_df=pd.read_csv(url2)


iris_df.info()
########
x_iris=iris_df.iloc[:,0:4]
y_iris=iris_df.iloc[:,4]

x_iris_sc=StandardScaler().fit_transform(x_iris)

x_iris_tr, x_irirs_tes, y_iris_tr, y_irirs_tes=train_test_split(x_iris_sc, y_iris,test_size=0.2, random_state=111)


ann1=keras.models.Sequential([
    keras.layers.Dense(128,  activation='tanh'),
    keras.layers.Dense(64,  activation='tanh'),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dense(3, activation='softmax')
])


opt=keras.optimizers.Adam(learning_rate=0.001)

ann1.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=keras.metrics.SparseCategoricalAccuracy() )


ann1.fit(x_iris_tr, y_iris_tr, epochs=40, validation_data=(x_irirs_tes,y_irirs_tes))

pred_test=np.argmax(ann1.predict(x_irirs_tes), axis=1)

########### regularizar
dor=0.5
sr = 0.1 ## por defecto
l2=keras.regularizers.l2(sr)


ann2=keras.models.Sequential([
    keras.layers.Dense(128,  activation='tanh', kernel_regularizer=l2),
    keras.layers.Dropout(dor),
    keras.layers.Dense(64,  activation='tanh'),
    keras.layers.Dropout(dor),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dropout(dor),
    keras.layers.Dense(3, activation='softmax')
])

opt=keras.optimizers.Adam(learning_rate=0.001)
ann2.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=keras.metrics.SparseCategoricalAccuracy() )
ann2.fit(x_iris_tr, y_iris_tr, epochs=40, validation_data=(x_irirs_tes,y_irirs_tes))

#### keras tunner ##########


hp=kt.HyperParameters()

def tun_model(hp):
    
    dor=hp.Float('DOR', min_value=0.1, max_value=0.6, step=0.1)
    opt=hp.Choice('opt', ['adam','sgd'])
    
    
    ann=keras.models.Sequential([
        keras.layers.Dense(128,  activation='tanh'),
        keras.layers.Dropout(dor),
        keras.layers.Dense(64,  activation='tanh'),
        keras.layers.Dropout(dor),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dropout(dor),
        keras.layers.Dense(3, activation='softmax')
    ])

    if opt== 'adam':
        opt=keras.optimizers.Adam(learning_rate=0.001)
    else:
        opt=keras.optimizers.SGD(learning_rate=0.001)
        
    ann.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=keras.metrics.SparseCategoricalAccuracy() )
   
    return ann


search_model=kt.RandomSearch(
    hypermodel=tun_model,
    hyperparameters=hp,
    objective=kt.Objective('val_sparse_categorical_accuracy', direction="max"),
    max_trials=20,
    overwrite=True,
    project_name="resultados",
    
)

search_model.search(x_iris_tr, y_iris_tr, epochs=10, validation_data=(x_irirs_tes,y_irirs_tes))
search_model.results_summary()

best_model=search_model.get_best_models(num_models=3)[0]
hps=search_model.get_best_hyperparameters(5)[0]
hps.values


best_model.build()
best_model.summary()
flay=best_model.layers[0]

###### medir #######


from sklearn import metrics

cm=metrics.confusion_matrix(y_irirs_tes, pred_test)
cm_disp=metrics.ConfusionMatrixDisplay(cm)
cm_disp.plot()

print(metrics.classification_report(y_irirs_tes,pred_test))




