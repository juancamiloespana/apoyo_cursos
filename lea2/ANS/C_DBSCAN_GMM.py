import seaborn as sns ### para los datos y para gráficar
import matplotlib.pyplot as plt
from sklearn import mixture ### modelos de clúster
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score ## indicador para consistencia de clúster
from kneed import KneeLocator ### para detectar analíticamente el cambio en la pendiente
from sklearn.model_selection import GridSearchCV


iris=sns.load_dataset('iris')

features=iris[['sepal_length', 'sepal_width']]
feat_sc=StandardScaler().fit_transform(features)

#### cómo identificar la forma

###

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




 
from sklearn.model_selection import GridSearchCV
 
param_grid={
        'n_components': [2, 3, 4, 5],  # Number of components/clusters
        'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Covariance type
        'n_init':[1,5,10]
         }
 
gs= GridSearchCV(gmm, param_grid=param_grid)
gs.fit(feat_sc)

gs.cv_results_
gs.best_score_
gs.best_params_

gmm.aic(feat_sc)

import pandas as pd

pd.set_option('display.max_colwidth', None)
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

ep2=eps*0.50

db= DBSCAN(eps=ep2, min_samples=4)
db.fit(feat_sc)

np.unique(db.labels_, return_counts=True)

sns.scatterplot(x=feat_sc[:,0], y=feat_sc[:,1], hue=db.labels_, palette="viridis")
plt.ylabel("sepal_length escalada")
plt.xlabel("sepal_width escalada")
plt.title("Gráfico")
plt.show()


### si se aumenta e eps  todos caen en un mismo cluster y disminuyen atípicos
### si se baja eps, aumentan atipicos y pueden 
# aparecer más cluster, está aumentando densidad porque en un espacio más reducido pido los mismos observaciones

## si aumento muestra, aumentan atipicos y se exige más densidad

### si disminuyo min sample se pueden disminuir atípicos, porque los que eran atípicos pueden formar cluster

#### si se quiere aumentar cluster:  bajar eps y min samples 
### si se quiere disminuir atípicos aumentar eps y mantener o bajar min samples




sil_sco=[]
ms=5
for ms in range(2,30):
    db= DBSCAN(eps=eps, min_samples=ms)
    db.fit(feat_sc)
    ss=silhouette_score(feat_sc,db.labels_)
    sil_sco.append(ss)


sns.lineplot(x=range(2,30), y=sil_sco)

pd.DataFrame(sil_sco)