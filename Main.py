import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)

severity_humidity = new_df.groupby('Severity')['Humidity(%)'].mean()
print(severity_humidity)

#Se quiseremos fazer correação
#humidity_severity = new_df[['Humidity(%)', 'Severity']]
#corr_coef = np.corrcoef(humidity_severity['Humidity(%)'], humidity_severity['Severity'])[0][1]
#print(corr_coef)



