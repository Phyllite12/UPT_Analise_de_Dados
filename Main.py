import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime as dt

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)

#Criação de um novo dataframe
df1 = pd.DataFrame(new_df)
#print(df1)

# Ordena a severidade por quartis (25%, 50%(mediana), 75%)
describSever = df.describe()
print(describSever)

#Acidentes Por estado
countsAciSta = new_df["State"].value_counts()
#print(countsAciSta)

#Acidentes nas diferentes timezones
timezone_counts = new_df['Timezone'].value_counts()
#print(timezone_counts)

severity_humidity = new_df.groupby('Severity')['Humidity(%)'].mean()
#print(severity_humidity)

#Se quiseremos fazer correação
humidity_severity = new_df[['Humidity(%)', 'Severity']]
corr_coef = np.corrcoef(humidity_severity['Humidity(%)'], humidity_severity['Severity'])[0][1]
#print(corr_coef)


sns.distplot(df['Severity'], kde=False, bins=10)
plt.show()

#sns.pairplot(df1[['Severity', 'Humidity(%)', 'Temperature(F)', 'Visibility(mi)']])
#plt.show()

#mapa de acidentes pela longitude e latitude
sample_df = df.sample(int(0.1 * len(df)))
sns.scatterplot(data=sample_df, x='Start_Lng', y='Start_Lat',size=0.001)
plt.show()

#HISTOGRAMA E CONTAGEM DO NºACIDENTES POR MÊS
new_df['Start_Time'] = pd.to_datetime(df['Start_Time']) # converte a coluna de data em um objeto de série de data
new_df['Month'] = new_df['Start_Time'].dt.month # extrai o mês de cada data na coluna "Start_Time"
print(new_df['Month'].value_counts()) #Contagem do nºde acidentes por mês
#plt.hist(new_df['Month'], bins=12)

#Adiciona títulos e rótulos aos eixos
#plt.title('Distribuição dos Acidentes de Carro ao Longo dos Meses')
#plt.xlabel('Mês')
#plt.ylabel('Número de Acidentes')

# exibe o histograma
#plt.show()



#BOXPLOTS------------------------------
#SEVERIDADE
# cria um boxplot da coluna "Severity"
plt.boxplot(new_df['Severity'])

# adiciona títulos e rótulos aos eixos
plt.title('Distribuição da Severidade dos Acidentes')
plt.ylabel('Severidade')

# exibe o boxplot
plt.show()

#HUMIDADE-------------------------------------------------
# cria um boxplot da coluna "Humidity(%)"
plt.boxplot(new_df['Humidity(%)'])

# adiciona títulos e rótulos aos eixos
plt.title('Distribuição da Severidade dos Acidentes')
plt.ylabel('Humidade(%)')

# exibe o boxplot
plt.show()

#TEMPERATURA-------------------------------------------
#cria um boxplot da coluna "Temperature(F)"
plt.boxplot(df['Temperature(F)'])

# adiciona títulos e rótulos aos eixos
plt.title('Distribuição da Temperatura no Momento dos Acidentes')
plt.ylabel('Temperatura (F)')

# exibe o boxplot
plt.show()

#transformar os dados categóricos em numéricos
df1['Sunrise_Sunset'] = df['Sunrise_Sunset'].map({'Day': 1, 'Night': 0})
df1['Crossing'] = df['Crossing'].map({'True': 1, 'False': 0})
df1['Traffic_Signal'] = df['Traffic_Signal'].map({'True': 1, 'False': 0})

#converter os dados de tempo para datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])
#calcular a diferenca entre o tempo de inicio e fim do acidente
difTime = df['End_Time'] - df['Start_Time']
df1['Acident_Duration'] = difTime

print(df1)
print(df1['Sunrise_Sunset'])