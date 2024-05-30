import pandas as pd
import numpy as np
import os
import openpyxl
from tqdm import tqdm


ruta= 'salidas\\ej\\'
list_f=os.listdir(ruta)

df_full=pd.read_csv('datos\\datos_nuevos_creditos_porc_pago.csv')


df_full["NoPaidPerc"].max()
df_full["nopago_presta"].max()


def aceptar(df):
    
    
    no_p=np.array(df['NoPaidLoan'])
    p_p=df['int_rc']

    conditions = [p_p <= 0.08,  (p_p < (no_p + 0.05))]

    choices = [1,1]
 

    default=0

    acepta=np.select(conditions, choices, default)
  
    
    df['acepta']=acepta 

    
    return df





equipo_ac=[]
aceptaron_ac=[]
total_prestado_ac=[]
total_pagado_ac=[]


for file in tqdm(list_f):
    
    print(ruta+file)
    file_path=ruta+file
    equipo=file.split('.')[0]
    equipo_ac.append(equipo)
    
    datos=pd.read_csv(file_path)
    
    datos['NoPaiFull']=df_full['NoPaidPerc']
    datos['NoPaidLoan']=df_full['nopago_presta']
   

 
    datos['NewLoanApplication']=df_full['NewLoanApplication']
    datos['monto_cobrar']=np.round((datos['int_rc']+1.15)*datos['NewLoanApplication'],1)
    datos=aceptar(datos)
    datos['pagado']=datos['monto_cobrar']*(1-datos['NoPaiFull'])

    file_res='salidas\\res\\resultados_'+file
    datos.to_csv(file_res, index=False)
    file_res2='salidas\\res\\resultados_'+equipo+'.xlsx'
    datos.to_excel(file_res2, index=False)

    total_prestado=(datos['NewLoanApplication']*datos['acepta']).sum()
    total_pagado=(datos['pagado']*datos['acepta']).sum()
    
   
    aceptaron=datos['acepta'].sum()
    aceptaron_ac.append(aceptaron)
    total_prestado_ac.append(np.round(total_prestado,1))
    total_pagado_ac.append(np.round(total_pagado,1))
    
    

resultados=pd.DataFrame()

resultados['equipo']=equipo_ac
resultados['aceptaron']=aceptaron_ac
resultados['rechazaron']=1058-resultados['aceptaron']
resultados['total_prestado']=total_prestado_ac
resultados['total_pagado']=total_pagado_ac
resultados['utilidad']= resultados['total_pagado'] -(resultados['total_prestado']*1.05)
resultados['margen']=resultados['utilidad']/resultados['total_prestado']

    


resultados.to_excel("salidas\\resultados.xlsx", index=False)

resultados.sort_values('utilidad', ascending=False)