
#install.packages("gamlss.data")
#install.packages("gamlss")

library(gamlss)
fitDist()

gamlss.dist::ZAPIG()

library(gamlss)
y <- rt(100, df=1)
m1<-fitDist(y, type="realline")
m1$fits
m1$failed
summary(m1)

m1$mu
m1$sigma
m1$sigma.coefficients
m2$nu

set.seed(123)
data <- r(100, mu = 5, nu = 2)

# Fit the ZAPIG distribution to the data
fit <- gamlss(data ~ 1, family = ZAPIG.family)

# Print the summary of the fitted distribution
summary(fit)

# Extract the estimated parameters of the ZAPIG distribution
best_params <- coef(fit)

# Generate new data from the ZAPIG distribution using the estimated parameters
new_data <- rzapig(100, mu = best_params["mu"], nu = best_params["nu"])

# Plot the original data and the newly generated data
par(mfrow = c(1, 2))
hist(data, main = "Original Data", xlab = "Value")
hist(new_data, main = "Generated Data", xlab = "Value")



datos=rZAPIG(100, mu=1, nu=0.4, sigma=1)

plot(datos, type="l")

m=fitDist(datos,type="counts")

#### solo esta parte cambia, en lugar de escribirlos con el .coefficients , se escriben solos, el .coefficient tiene una transformación, y porbablemente poreso salía un error cuando generabamos la distribución con los parámetros calculados.

mu_f=m$mu
sig_f=m$sigma
nu_f=m$nu

datos_nuevo=rZAPIG(n=720,mu =mu_f, sigma=sig_f, nu=nu_f )


#############################################################
#####ejemplo amcor ############################################

datos=read.csv("datos\\demanda.csv", sep=';')


dist=fitDist(datos$Demanda_pallete, type="count", trace=T)

mu=dist$mu
dist$fits
dist




plot(datos_nuevos)

fit <- gamlss(datos$Demanda_pallete ~ 1, family =  ZINBI(mu.link = 'identity') )

sigma=exp(fit$sigma.coefficients)
nu=plogis(fit$nu.coefficients)
mu=fit$mu.coefficients

datos_nuevos=rZINBI(700, mu=70, sigma = sigma*0.8, nu=nu*0.8)

max(datos_nuevos)
hist(datos_nuevos)

hist(datos$Demanda_pallete, breaks = 20)


write.csv(datos_nuevos, "datos_nuevos.csv", row.names = F)
