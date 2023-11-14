################Ejercicio profesor #############

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.mixture import GaussianMixture


url='https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/customer_personality.csv'
data = pd.read_csv(url, sep="\t")




data.info()

data.dropna(inplace=True)

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
data.sort_values('Dt_Customer')
f=max(data['Dt_Customer']) 


data['dias']
dias=f-data['Dt_Customer'] 
dias.dt.days

data['Dt_Customer'] = dias.dt.days

data['Marital_Status'].value_counts()
data['Education'].value_counts()

data['Marital_Status']=data['Marital_Status'].replace({'Alone':'Single','Absurd':"Single", 'YOLO':'Single'})
data['Marital_Status'].value_counts()


data_dum=pd.get_dummies(data,['Education', 'Marital_status'])

data_dum.info()
############### Ejercicio estudiantes ##################

feat_sc=StandardScaler().fit_transform(data_dum)


pca=PCA(n_components=2)
pca.fit(feat_sc)

pca.explained_variance_ratio_.sum()

rd=pca.transform(feat_sc)

sns.scatterplot(x=rd[:,0], y=rd[:,1])


import numpy as np
from sklearn.metrics import silhouette_score


AIC_clust=[]
score_cum=[]
sil=[]



for i in range(2,10):
    
    gmm=GaussianMixture(n_components=i, n_init=10, covariance_type="full")
    gmm.fit(feat_sc)
    aic=gmm.aic(feat_sc)
    AIC_clust.append(aic)
    score_ind=gmm.score(feat_sc)
    score_cum.append(score_ind)
    sil_ind=silhouette_score(feat_sc, gmm.predict(feat_sc))
    sil.append(sil_ind)
    


sns.lineplot(x=np.arange(2,10), y =sil)

len(AIC_clust)
len(np.arange(2,10))
