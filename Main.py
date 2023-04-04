import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)

df1 = df.DataFrame(df) 
print(df1)

df2 = df.describe()# Ordena por quartis (25%, 50%(mediana), 75%)
print(df2)

df3 = df.T #Transpoem a tabela
print(df3)

df4 =  df.size #diz o tamanho
print(df4)

df5 = df.mean() #diz a media de acidentes por ano falta fazer o filtro
print("Media de acidentes por ano",df5)