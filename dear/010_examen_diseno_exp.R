
library(datarium)
library(tidyverse)
library(rstatix)

data(datarium)


####Ejercicio 1 ###########

data("depression")

depression[,3] = depression[,3]


#### Fin código para cargar base de datos ###############
dep=depression%>%gather(key='time', value="dep", t0, t1, t2, t3)
dep$time=as.factor(dep$time)

attach(dep)
f=dep~time+Error(id/time)

anova1=anova_test(data=dep, f)
anova1

anova2=aov(f, data=dep)
summary(anova2)




####Ejercicio 2 ###########

data("depression")

set.seed(123)

depression[,3] = depression[,3]*runif(1,0,0.5)


#### Fin código para cargar base de datos ###############
dep=depression%>%gather(key='time', value="dep", t0, t1, t2, t3)
dep$time=as.factor(dep$time)

attach(dep)
f=dep~time+Error(id/time)

anova1=anova_test(data=dep, f)
anova1

anova2=aov(f, data=dep)
summary(anova2)




####Ejercicio 3 ###########

data("depression")

set.seed(9)

depression[,4] = depression[,4]*runif(1,0,0.5)


#### Fin código para cargar base de datos ###############
dep=depression%>%gather(key='time', value="dep", t0, t1, t2, t3)
dep$time=as.factor(dep$time)

attach(dep)
f=dep~time+Error(id/time)

anova1=anova_test(data=dep, f)
anova1

anova2=aov(f, data=dep)
summary(anova2)



#########################factoruial 1 


  
  
  
  url='https://raw.githubusercontent.com/juancamiloespana/DEAR_JCE/master/data/tiempo_prod.csv'

tiempo_prod=read.csv(url)



tiempo_prod$operador=as.factor(tiempo_prod$operador)

tiempo_prod$maquina=as.factor(tiempo_prod$maquina)

tiempo_prod$proceso=as.factor(tiempo_prod$proceso)

tiempo_prod$jornada=as.factor(tiempo_prod$jornada)



mod2=aov(data=tiempo_prod, tiempo_prod~maquina*proceso)

summary(mod2)  ## para punto 2, 3 y 4

model.tables(x=mod2,type='effects') ## para punto 1






