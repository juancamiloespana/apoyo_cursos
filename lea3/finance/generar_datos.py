import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

import seaborn as sns
from sklearn import metrics

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows and columns
n_rows = 10000


# Generate random data for the DataFrame
data = {
    'ID': range(1, n_rows + 1),
    'CreditScore': np.random.randint(300, 850, n_rows),
    'DebtRatio': np.random.uniform(0, 1, n_rows),
    'Assets': np.random.randint(20000, 200000, n_rows),
    'Age': np.random.randint(18, 90, n_rows),
    'NumberOfDependents': np.random.randint(0, 10, n_rows),
    'NumberOfOpenCreditLinesAndLoans': np.random.randint(0, 20, n_rows),
    'MonthlyIncome': np.random.randint(1000, 20000, n_rows),
    'NumberOfTimesPastDue': np.random.randint(0, 20, n_rows),
    'EmploymentLength': np.random.randint(0, 30, n_rows),  # Number of years employed
    'HomeOwnership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_rows),  # Type of home ownership
    'Education': np.random.choice(['High School', 'Bachelor', 'Masters', 'PhD'], n_rows),  # Education level
    'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_rows),  # Marital status
    'YearsAtCurrentAddress': np.random.randint(0, 30, n_rows),  # Number of years at current address
}

# Create DataFrame
df = pd.DataFrame(data)
df2=pd.get_dummies(df)
x=StandardScaler().fit_transform(df2)
x.shape


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(22,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

p=model.predict(x)
p2=p.reshape(-1)/2
y=p2
y2=np.where(y <= 0.15, 0, y)

sns.histplot(y=y2)



l=tf.keras.losses.MeanSquaredError()
m=tf.keras.metrics.MeanSquaredError()
model.compile(loss=l, metrics=m, optimizer="adam")

model.fit(x,y2,epochs=10)
pred=model.predict(x)
pred2=pred.reshape(-1)


d=metrics.PredictionErrorDisplay.from_predictions(y2,pred2, kind='actual_vs_predicted' )
d.plot()

df=df.drop('Nopaid_perc', axis=1)

df['NoPaidPerc']=y2


# Display first few rows of the DataFrame
print(df.head())

# Data description
data_description = """
Data Description:

- ID: Unique identifier for each record.
- CreditScore: The credit score of the individual.
- DebtRatio: The ratio of debt to total assets.
- Assets: Total assets of the individual.
- Age: The age of the individual.
- NumberOfDependents: The number of dependents of the individual.
- NumberOfOpenCreditLinesAndLoans: The number of open credit lines and loans.
- MonthlyIncome: The monthly income of the individual.
- NumberOfTimesPastDue: The number of times the individual has been past due.
- EmploymentLength: Number of years employed.
- HomeOwnership: Type of home ownership (Rent, Own, Mortgage).
- Education: Education level (High School, Bachelor, Masters, PhD).
- MaritalStatus: Marital status (Single, Married, Divorced, Widowed).
- YearsAtCurrentAddress: Number of years at current address.
- NoPaidPerc: Proportion of the payments that consumer has not made in all their credits.
"""

# Write the data description to a text file
with open('data_description.txt', 'w') as file:
    file.write(data_description)
    
    

df_new=df[df['NoPaidPerc']<0.17]

df_new.to_csv('datos_nuevos_creditos_porc_pago.csv', index=False)
df_nwe0=df_new

df_new['ID']=range(10001,  11059)

df_new.reset_index(inplace=True, drop=True)
c


df.to_csv('datos_historicos.csv', index=False)


prestamos=np.round(np.random.rand(1058)*20,0)+ 2
prestamos.min()

df_new['NewLoanApplication']=df_new['MonthlyIncome']*prestamos
df_new.to_csv('datos_nuevos_creditos.csv', index=False)


###########################Cargar daots
df_new0
