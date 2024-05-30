import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

import seaborn as sns
from sklearn import metrics


df_new=pd.read_csv('datos\\datos_nuevos_creditos_porc_pago.csv')

total_prestado=df_new['NewLoanApplication'].sum()


###############################
#### interes tottal ##########
##############################


#### confirmar que de igual que el calculado en la linea anterior
pago_cliente=df_new['NewLoanApplication']*df_new['NoPaidPerc']
total_pago=pago_cliente.sum()
porc_no_pago=total_pago/total_prestado



##### el monto con costo va a ser el 1-porc de no pago

prop_pagado=1-porc_no_pago
meta_pago =df_new['NewLoanApplication'].sum()*1.1
monto_cobrar= meta_pago/prop_pagado 
interes_total=monto_cobrar/total_prestado ## este valor da en total pero no aplicado individualmente




########################################



meta_cliente=df_new['NewLoanApplication']*1.15
cobro_cliente= meta_cliente/(1-df_new['NoPaidPerc'])
interes_cliente= cobro_cliente/df_new['NewLoanApplication']

lim_cliente=interes_cliente.min()





##### entrenar modelo ###################

df_hist=pd.read_csv('datos_historicos.csv')
df2=pd.get_dummies(df_hist)
y=df2['NoPaidPerc']
y=np.array(y)
sns.histplot(x=y)
x_dummies=df2.drop('NoPaidPerc',axis=1)
sc=StandardScaler().fit(x_dummies)
x=sc.transform(x_dummies)
x.shape


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(22,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


l=tf.keras.losses.MeanSquaredError()
m=tf.keras.metrics.RootMeanSquaredError()
model.compile(loss=l, metrics=[m], optimizer="adam")
model.fit(x,y,epochs=10, validation_split=0.2)

model.fit(x,y,epochs=10)
model.save('modelo.keras')


x_new=df_new.drop(['NewLoanApplication','NoPaidPerc'],axis=1)
x_new_dummies=pd.get_dummies(x_new)
x_n=sc.transform(x_new_dummies)

pred=model.predict(x_n)
pred=np.array(pred)

pagado=df_new['NewLoanApplication']*1.15
df_new['PredNoPago']=pred
monto_cob=pagado/(1-df_new['PredNoPago'])
int_rc=(monto_cob-pagado)/df_new['NewLoanApplication']


monto_cob_R=pagado/(1-df_new['NoPaidPerc'])
nopago_prest=(monto_cob_R-pagado)/df_new['NewLoanApplication']

df_new["nopago_presta"]=nopago_prest

#df_new.to_csv('datos\\datos_nuevos_creditos_porc_pago.csv')

sns.histplot(x=nopago_prest)
sns.histplot(x=int_rc)
sns.histplot(x=df_new['PredNoPago'])
sns.histplot(x=df_new['NoPaidPerc'])


error=df_new['NoPaidPerc']-df_new['PredNoPago']
df_new["error"]=error

sns.histplot(x=error)
np.mean(np.abs(error))

df_new['NoPaidPerc'].max()
df_new['PredNoPago'].max()


sns.histplot(x=df_new['PredNoPago'])

no_p=np.array(df_new['NoPaidPerc'])
p_p=df_new['PredNoPago']

conditions = [
    p_p <= 0.05,
    p_p >= 0.20,
    (p_p < (no_p + 0.02))
]

choices = [
    1,
    0,
    1
]
choices2 = [
    1,
    3,
    2
]

default=0

acepta=np.select(conditions, choices, default)
cond=np.select(conditions, choices2, default)

np.unique(acepta, return_counts=True)
np.unique(cond, return_counts=True)


profesor1=pd.DataFrame()

profesor1['ID']= df_new['ID']
profesor1['int_rc']=int_rc+0.02

profesor1.to_csv('salidas\\ej\\profesor1.csv', index=False)