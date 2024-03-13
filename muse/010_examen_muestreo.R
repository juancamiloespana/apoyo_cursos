###############Ejericio 1 ################

library(TeachingSampling) 

data("Lucy")

attach(Lucy)



############primer punto ####################

N=nrow(Lucy)
n=400

sd= 18 ### esta en millones y está definido teoricamente
nsd=2 ### una confianza del 95%
B= 3   ####si es pequeño muestra es muy grande

D= (B/nsd)^2  
n= N*(sd^2)/((N-1)*D +sd^2)
round(n)

#############################################
## segundo punto
################################################


n=400
set.seed(123)

sam= S.SI(N,n) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SI(N,n, sam_datos)
estimaciones



#############systematico



n=60
a=round(N/n)
set.seed(123)
sam= S.SY(N,a) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SY(N,a, sam_datos)
estimaciones

estimaciones[1,2]/N


#########estratificado

n=400
Nst=table(Lucy$SPAM) ## tamaño por estrato
alphast=(Nst/N) ###proporciones por estrato
nst= round(alphast*n) ### muestra por estratificado


#####
set.seed(123)
sam3=S.STSI(Lucy$SPAM, Nst, nst)
sam_datos3=Lucy[sam3,c("Taxes", "SPAM")]
sam_datos3=data.frame(sam_datos3)


estimaciones3=E.STSI(sam_datos3$SPAM, Nst, nst, sam_datos3$Taxes)
estimaciones3


################################################
####### semilla 321 #############################
################################################

N=nrow(Lucy)
n=400

sd= 18 ### esta en millones y está definido teoricamente
nsd=3 ### una confianza del 95%
B= 5  ####si es pequeño muestra es muy grande

D= (B/nsd)^2  
n= N*(sd^2)/((N-1)*D +sd^2)
round(n)

#############################################
## segundo punto
################################################


n=400
set.seed(321)

sam= S.SI(N,n) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SI(N,n, sam_datos)
estimaciones



#############systematico



n=50
a=round(N/n)
set.seed(321)
sam= S.SY(N,a) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SY(N,a, sam_datos)
estimaciones

estimaciones[1,2]/N


#########estratificado

n=200
Nst=table(Lucy$SPAM) ## tamaño por estrato
alphast=(Nst/N) ###proporciones por estrato
nst= round(alphast*n) ### muestra por estratificado


#####
set.seed(321)
sam3=S.STSI(Lucy$SPAM, Nst, nst)
sam_datos3=Lucy[sam3,c("Taxes", "SPAM")]
sam_datos3=data.frame(sam_datos3)


estimaciones3=E.STSI(sam_datos3$SPAM, Nst, nst, sam_datos3$Taxes)
estimaciones3




################################################
####### semilla 999 #############################
################################################

N=nrow(Lucy)
n=400

sd= 22 ### esta en millones y está definido teoricamente
nsd=3 ### una confianza del 95%
B= 5  ####si es pequeño muestra es muy grande

D= (B/nsd)^2  
n= N*(sd^2)/((N-1)*D +sd^2)
round(n)

#############################################
## segundo punto
################################################


n=200
set.seed(999)

sam= S.SI(N,n) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SI(N,n, sam_datos)
estimaciones



#############systematico



n=200
a=round(N/n)
set.seed(999)
sam= S.SY(N,a) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SY(N,a, sam_datos)
estimaciones

estimaciones[1,2]/N


#########estratificado

n=200
Nst=table(Lucy$SPAM) ## tamaño por estrato
alphast=(Nst/N) ###proporciones por estrato
nst= round(alphast*n) ### muestra por estratificado


#####
set.seed(999)
sam3=S.STSI(Lucy$SPAM, Nst, nst)
sam_datos3=Lucy[sam3,c("Taxes", "SPAM")]
sam_datos3=data.frame(sam_datos3)


estimaciones3=E.STSI(sam_datos3$SPAM, Nst, nst, sam_datos3$Taxes)
estimaciones3




################################################
####### semilla 7 #############################
################################################

sem=7

N=nrow(Lucy)
n=400

sd= 22 ### esta en millones y está definido teoricamente
nsd=2 ### una confianza del 95%
B= 1  ####si es pequeño muestra es muy grande

D= (B/nsd)^2  
n= N*(sd^2)/((N-1)*D +sd^2)
round(n)

#############################################
## segundo punto
################################################


n=200
set.seed(sem)

sam= S.SI(N,n) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SI(N,n, sam_datos)
estimaciones



#############systematico



n=200
a=round(N/n)
set.seed(sem)
sam= S.SY(N,a) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SY(N,a, sam_datos)
estimaciones

estimaciones[1,2]/N


#########estratificado

n=200
Nst=table(Lucy$SPAM) ## tamaño por estrato
alphast=(Nst/N) ###proporciones por estrato
nst= round(alphast*n) ### muestra por estratificado


#####
set.seed(sem)
sam3=S.STSI(Lucy$SPAM, Nst, nst)
sam_datos3=Lucy[sam3,c("Taxes", "SPAM")]
sam_datos3=data.frame(sam_datos3)


estimaciones3=E.STSI(sam_datos3$SPAM, Nst, nst, sam_datos3$Taxes)
estimaciones3



################################################
####### semilla 0 #############################
################################################

sem=0

N=nrow(Lucy)
n=400

sd= 10 ### esta en millones y está definido teoricamente
nsd=2 ### una confianza del 95%
B= 1  ####si es pequeño muestra es muy grande

D= (B/nsd)^2  
n= N*(sd^2)/((N-1)*D +sd^2)
round(n)

#############################################
## segundo punto
################################################


n=200
set.seed(sem)

sam= S.SI(N,n) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SI(N,n, sam_datos)
estimaciones



#############systematico



n=200
a=round(N/n)
set.seed(sem)
sam= S.SY(N,a) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SY(N,a, sam_datos)
estimaciones

estimaciones[1,2]/N


#########estratificado

n=200
Nst=table(Lucy$SPAM) ## tamaño por estrato
alphast=(Nst/N) ###proporciones por estrato
nst= round(alphast*n) ### muestra por estratificado


#####
set.seed(sem)
sam3=S.STSI(Lucy$SPAM, Nst, nst)
sam_datos3=Lucy[sam3,c("Taxes", "SPAM")]
sam_datos3=data.frame(sam_datos3)


estimaciones3=E.STSI(sam_datos3$SPAM, Nst, nst, sam_datos3$Taxes)
estimaciones3



################################################
####### semilla 2 #############################
################################################

sem=2

N=nrow(Lucy)
n=400

sd= 15 ### esta en millones y está definido teoricamente
nsd=2 ### una confianza del 95%
B= 6  ####si es pequeño muestra es muy grande

D= (B/nsd)^2  
n= N*(sd^2)/((N-1)*D +sd^2)
round(n)

#############################################
## segundo punto
################################################


n=200
set.seed(sem)

sam= S.SI(N,n) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SI(N,n, sam_datos)
estimaciones



#############systematico



n=200
a=round(N/n)
set.seed(sem)
sam= S.SY(N,a) ### este me selecciona las filas para la muestra
sam_datos= Lucy[sam, 'Taxes'] 
estimaciones=E.SY(N,a, sam_datos)
estimaciones

estimaciones[1,2]/N


#########estratificado

n=200
Nst=table(Lucy$SPAM) ## tamaño por estrato
alphast=(Nst/N) ###proporciones por estrato
nst= round(alphast*n) ### muestra por estratificado


#####
set.seed(sem)
sam3=S.STSI(Lucy$SPAM, Nst, nst)
sam_datos3=Lucy[sam3,c("Taxes", "SPAM")]
sam_datos3=data.frame(sam_datos3)


estimaciones3=E.STSI(sam_datos3$SPAM, Nst, nst, sam_datos3$Taxes)
estimaciones3
