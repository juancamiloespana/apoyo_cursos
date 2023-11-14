from sklearn import mixture
import seaborn as sns
from sklearn.preprocessing import StandardScaler



iris=sns.load_dataset('iris')

features=iris[['sepal_length', 'sepal_width']]
feat_sc=StandardScaler().fit_transform(features)

#### cómo identificar la forma

sns.scatterplot(x=feat_sc[:,0],y=feat_sc[:,1])

gmm=mixture.GaussianMixture(n_components=3,
                            covariance_type='full', 
                            n_init=10,
                            init_params="random"
                            )

gmm.fit(feat_sc)

gmm.aic(feat_sc)
gmm.score(feat_sc)


gmm2=mixture.GaussianMixture(n_components=3,
                            covariance_type='tied', 
                            n_init=10,
                            init_params="random",
                            )

 gmm2.fit(feat_sc)
 gmm2.aic(feat_sc)
 gmm.aic(feat_sc)
 gmm2.score(feat_sc)
 gmm.score(feat_sc)
 
 from sklearn.model_selection import GridSearchCV
 
 param_grid={
        'n_components': [2, 3, 4, 5],  # Number of components/clusters
        'covariance_type': ['full', 'tied', 'diag', 'spherical']  # Covariance type
        'n_init'=[1,5,10],
        'init_params'=['random']
        
        }
 
 gs= GridSearchCV(gmm2, param_grid=param_grid)
 gs.fit(feat_sc)

 gs.cv_results_
 gs.best_score_

import pandas as pd
df=pd.DataFrame(gs.cv_results_)
df.sort_values(['mean_test_score'], ascending=False)[['params','mean_test_score']]





########dbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
#### determinar epsilon con vecino más cercano

knn=NearestNeighbors(n_neighbors=50)
knn.fit(feat_sc)

distance, *_ =knn.kneighbors(feat_sc)
distance[1]

distancia_k_vecinos=np.mean(distance, axis=0)
len(distancia_k_vecinos)

rango= np.arange(1,len(distancia_k_vecinos)+1)
plt.plot(rango, distancia_k_vecinos)

from kneed import KneeLocator

kl=KneeLocator(x=rango, y=distancia_k_vecinos, curve="concave", direction="increasing")

eps=distancia_k_vecinos[kl.elbow]

#### determinar min_samples
from sklearn.cluster import DBSCAN


sil_sco=[]
ms=5
for ms in range(2,30):
    db= DBSCAN(eps=eps, min_samples=ms)
    db.fit(feat_sc)
    ss=silhouette_score(feat_sc,db.labels_)
    sil_sco.append(ss)


sns.lineplot(x=range(2,30), y=sil_sco)

pd.DataFrame(sil_sco)

ep2=eps

db= DBSCAN(eps=ep2*0.5, min_samples=7)
db.fit(feat_sc)

db.labels_

sns.scatterplot(x=feat_sc[:,0], y=feat_sc[:,1], hue=db.labels_, palette="viridis")
plt.ylabel("sepal_length escalada")
plt.xlabel("sepal_width escalada")
plt.title("Gráfico")
plt.show()