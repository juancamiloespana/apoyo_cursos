import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import decomposition as deco
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


iris=sns.load_dataset("iris")
feat=iris.iloc[:,0:4]
iris_sc=StandardScaler().fit_transform(feat)

pca=PCA(n_components=4)

pca.fit(iris_sc)

pca.components_  ### vectores propios lambdas
pca.explained_variance_ratio_.sum() ### valores propios alphas


iris_pca=pca.transform(iris_sc) ### componentes principales L

df_iris_pca=pd.DataFrame(iris_pca, columns=['pca1','pca2','pca3','pca4'])
### para cada valor ####
np.dot(iris_sc[0],pca.components_[0]) ### para calucular valores de los l

###### graficar datos

sns.scatterplot(x=df_iris_pca['pca1'], y=df_iris_pca['pca2'],hue=iris['species'])

var=np.cumsum(pca.explained_variance_ratio_)

from kneed import KneeLocator

kl=KneeLocator(x=np.arange(1,5), y=var,curve="concave", direction="increasing")
kl.elbow
sns.lineplot(x=np.arange(1,5), y=var) 

pca.explained_variance_
pca.explained_variance_ratio_
pca.noise_variance_
pca.components_
iris_pca.var(axis=0)
pca.get_covariance()
np.sum(pca.get_covariance())


pca.explained_variance_/iris_sc.var(axis=0).sum()

#####


fa=deco.FactorAnalysis(n_components=2)
fa.components_
fa.fit(iris_sc)
np.sum(fa.get_covariance(), axis=0)
fa.noise_variance_.sum()

np.dot(iris_sc[0], fa.components_[0].T)

fa.mean_[0] + fa.noise_variance_[0]

iris_sc[0]
fa.transform(iris_sc)[0]+fa.mean_[0] + fa.noise_variance_[0]

np.dot(fa.transform(iris_sc)[0], fa.components_)
#####

from factor_analyzer import FactorAnalyzer

fa2=FactorAnalyzer(n_factors=2)
fa2.fit(iris_sc)
fa2.loadings_
fa2.transform(iris_sc)[100] ## genera fa2.weights
fa2.weights_



fa2.get_eigenvalues()
fa2.get_communalities()

fa2.get_factor_variance()
fa2.get_params()
iris_sc[1]
np.dot(iris_sc ,fa2.loadings_)[0]
fa2.rotation_matrix_
fa2.transform(iris_sc)[0]
fa2.mean_

fa2.transform(iris_sc)[100] ## genera latentes
iris_sc[1]
np.dot(fa2.loadings_,fa2.transform(iris_sc)[1]) ## estima normales con lantentes
np.dot(iris_sc,fa2.weights_)[100] ## calcula latentes
fa2.transform(iris_sc)[100]






# x= W*L +E

# w=loadings_, L = transform(data)
