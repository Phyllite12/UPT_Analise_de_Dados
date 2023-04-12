import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)

#Acidentes Por estado
countsAciSta = new_df["State"].value_counts()
#print(countsAciSta)

#Acidentes nas diferentes timezones
timezone_counts = new_df['Timezone'].value_counts()
#print(timezone_counts)

#HISTOGRAMA E CONTAGEM DO NºACIDENTES POR MÊS
new_df['Start_Time'] = pd.to_datetime(df['Start_Time']) # converte a coluna de data em um objeto de série de data
new_df['Month'] = new_df['Start_Time'].dt.month # extrai o mês de cada data na coluna "Start_Time"
print(new_df['Month'].value_counts()) #Contagem do nºde acidentes por mês
plt.hist(new_df['Month'], bins=12)

# adiciona títulos e rótulos aos eixos
plt.title('Distribuição dos Acidentes de Carro ao Longo dos Meses')
plt.xlabel('Mês')
plt.ylabel('Número de Acidentes')

# exibe o histograma
plt.show()