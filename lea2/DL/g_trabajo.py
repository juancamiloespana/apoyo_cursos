import pandas as pd
import numpy as np
import random
df=pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/LEA2/master/_data/credit_card_clients.csv')


df2=df.drop('PURCHASES', axis=1)

df2.to_csv('credit_card_clients2.csv', index=False)



