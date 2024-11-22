import pandas as pd
import sqlite3 as sql
import numpy as np

###### modificar fechas de retiro

retire=pd.read_csv('retirement_info.csv', sep=',')

pd.set_option('display.max_rows', 1000)
retire['retirementDate'] = pd.to_datetime(retire['retirementDate'])
retire['retirementDate'].dt.year.value_counts().sort_index()

retire.drop(["Unnamed: 0.1","Unnamed: 0"], axis=1, inplace=True)
retire.to_csv('retirement_info.csv', index=False)

retired_2015 = retire[retire['retirementDate'].dt.year == 2015]['EmployeeID']
retired_2016 = retire[retire['retirementDate'].dt.year == 2016]['EmployeeID']

#####################################################################
###### modificar encuestas de satisfacción ###########################################
##################################################################

survey=pd.read_csv('employee_survey_data.csv', sep=',')
survey.drop(["Unnamed: 0"], axis=1, inplace=True)

condition1 = (survey['EmployeeID'].isin(retired_2015 )) 
condition1.value_counts()

condition2 = (survey['EmployeeID'].isin(retired_2016 )) & (survey['DateSurvey'] == '2015-12-31') 
condition2.value_counts()

# Add 4410 to the common EmployeeIDs in the survey DataFrame
survey.loc[condition1, 'EmployeeID'] += 4410
survey.loc[condition2, 'EmployeeID'] += 4410





survey.to_csv('employee_survey_data.csv', index=False)


#####################################################################
###### General data ###########################################
##################################################################

general=pd.read_csv('general_data.csv', sep=',')

general.drop(["Unnamed: 0"], axis=1, inplace=True)

condition1 = (general['EmployeeID'].isin(retired_2015 )) 
condition1.value_counts()

condition2 = (general['EmployeeID'].isin(retired_2016 )) & (general['InfoDate'] == '2016-12-31') 
condition2.value_counts()

# Add 4410 to the common EmployeeIDs in the survey DataFrame
general.loc[condition1, 'EmployeeID'] += 4410
general.loc[condition2, 'EmployeeID'] += 4410



general.to_csv('general_data.csv', index=False)


########################## manager survey data ##########

manager1= pd.read_csv('manager_survey.csv', sep=',')

manager1.drop(["Unnamed: 0"], axis=1, inplace=True)

condition1 = (manager1['EmployeeID'].isin(retired_2015 )) 
condition1.value_counts()

condition2 = (manager1['EmployeeID'].isin(retired_2016  )) & (manager1['SurveyDate'] == '2015-12-31') 
condition2.value_counts()

# Add 4410 to the common EmployeeIDs in the survey DataFrame
manager1.loc[condition1, 'EmployeeID'] += 4410
manager1.loc[condition2, 'EmployeeID'] += 4410


manager1.to_csv('manager_survey.csv', index=False)