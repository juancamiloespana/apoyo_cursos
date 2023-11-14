
library(forecast)
library(fpp2)


findfrequency(h02)
Pacf(euretail)
autoplot(h02)
frequency(h02)

start(h02)
end(h02)
length(h02)
drug1=ts(h02, start=start(h02), end=c(2005,12), frequency=12)
drug2=window(h02, start=c(2006,1), frequency=12)

drug1_df=data.frame(dem=drug1)
drug2_df=data.frame(dem=drug2)

write.csv(drug1_df, 'data\\demanda_medicamento.csv', row.names = F)
write.csv(drug2_df, 'data\\demanda_medicamento2.csv', row.names = F)

autoplot(drug1)



drug1=ts(drug1_df$dem, start=c(1991,7), frequency=12)
start(drug1)
end(drug1)
