import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)

#Criação de um novo dataframe
df1 = pd.DataFrame(new_df)
print(df1)

#Acidentes Por estado
countsAciSta = new_df["State"].value_counts()
print(countsAciSta)

#Acidentes nas diferentes timezones
timezone_counts = new_df['Timezone'].value_counts()
print(timezone_counts)

severity_humidity = new_df.groupby('Severity')['Humidity(%)'].mean()
print(severity_humidity)

#Se quiseremos fazer correação
humidity_severity = new_df[['Humidity(%)', 'Severity']]
corr_coef = np.corrcoef(humidity_severity['Humidity(%)'], humidity_severity['Severity'])[0][1]
print(corr_coef)
