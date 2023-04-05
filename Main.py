import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)

#Acidentes Por estado
countsAciSta = new_df["State"].value_counts()
print(countsAciSta)

#Acidentes nas diferentes timezones
timezone_counts = new_df['Timezone'].value_counts()
print(timezone_counts)