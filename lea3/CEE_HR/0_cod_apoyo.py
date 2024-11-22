import pandas as pd
import sqlite3 as sql
import numpy as np

###### modificar fechas de retiro

retire=pd.read_csv('retirement_info.csv', sep=',')

pd.set_option('display.max_rows', 1000)
retire['retirementDate'] = pd.to_datetime(retire['retirementDate'])
retire['retirementDate'].dt.year.value_counts().sort_index()


retire['retirementDate'].iloc[650:]=retire['retirementDate'].iloc[650:].apply(lambda x: x.replace(year=2015))


retire.to_csv('retirement_info.csv')

#####################################################################
###### modificar encuestas de satisfacción ###########################################
##################################################################

survey=pd.read_csv('employee_survey_data.csv', sep=',')
survey.info()
survey['DateSurvey'] = '2015-12-31'

survey2=survey.copy()
survey2['DateSurvey']= '2016-12-31'

survey_final=pd.concat([survey, survey2], axis=0)

survey_final['DateSurvey'].value_counts()



survey_final.to_csv('employee_survey_data.csv')


#####################################################################
###### General data ###########################################
##################################################################

general=pd.read_csv('general_data.csv', sep=';')
general['InfoDate']= '2015-12-31'

general2=general.copy()
general2['InfoDate']='2016-12-31'

general_final=pd.concat([general, general2], axis=0)

general_final.to_csv('general_data.csv')


########################## manager survey data ##########

manager1= pd.read_csv('manager_survey_data.csv', sep=',')
manager1['SurveyDate']="2015-12-31"

manager2=manager1.copy()
manager2['SurveyDate']="2016-12-31"

manager=pd.concat([manager1, manager2], axis=0)

manager['SurveyDate'].value_counts()

manager.to_csv('manager_survey.csv')