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

#converter a temperatura de fahrenheit para celsius
df1['Temperature(C)'] = (df['Temperature(F)'] - 32) * 5/9

#converter a distancia de milhas para metros
df1['Distance(M)'] = (df['Distance(mi)'] * 1609.34)

# Ordena a severidade por quartis (25%, 50%(mediana), 75%)
describSever = df.describe()
print(describSever)

#Acidentes Por estado
countsAciSta = df1["State"].value_counts()
#print(countsAciSta)

#Acidentes nas diferentes timezones
timezone_counts = df1['Timezone'].value_counts()
#print(timezone_counts)

severity_humidity = df1.groupby('Severity')['Humidity(%)'].mean()
#print(severity_humidity)

#Se quiseremos fazer correação
humidity_severity = df1[['Humidity(%)', 'Severity']]
corr_coef = np.corrcoef(humidity_severity['Humidity(%)'], humidity_severity['Severity'])[0][1]
#print(corr_coef)


sns.distplot(df1['Severity'], kde=False, bins=10)
plt.show()

#cria grafico de dispersão entre a severidade e a temperatura em fahrenheit
#sns.pairplot(df1[['Severity', 'Humidity(%)', 'Temperature(F)', 'Visibility(mi)']])
#plt.show()



#HISTOGRAMA E CONTAGEM DO NºACIDENTES POR MÊS
df1['Start_Time'] = pd.to_datetime(df['Start_Time']) # converte a coluna de data em um objeto de série de data
df1['Month'] = df1['Start_Time'].dt.month # extrai o mês de cada data na coluna "Start_Time"
print(df1['Month'].value_counts()) #Contagem do nºde acidentes por mês
#plt.hist(new_df['Month'], bins=12)

#Adiciona títulos e rótulos aos eixos
plt.title('Distribuição dos Acidentes de Carro ao Longo dos Meses')
plt.xlabel('Mês')
plt.ylabel('Número de Acidentes')

# exibe o histograma
#plt.show()


#BOXPLOTS------------------------------
#SEVERIDADE
# cria um boxplot da coluna "Severity"
plt.boxplot(df1['Severity'])

# adiciona títulos e rótulos aos eixos
plt.title('Distribuição da Severidade dos Acidentes')
plt.ylabel('Severidade')

# exibe o boxplot
plt.show()

#HUMIDADE-------------------------------------------------
# cria um boxplot da coluna "Humidity(%)"
plt.boxplot(df1['Humidity(%)'])

# adiciona títulos e rótulos aos eixos
plt.title('Distribuição da Severidade dos Acidentes')
plt.ylabel('Humidade(%)')

# exibe o boxplot
plt.show()

#TEMPERATURA-------------------------------------------
#cria um boxplot da coluna "Temperature(C)"
plt.boxplot(df1['Temperature(C)'])

# adiciona títulos e rótulos aos eixos
plt.title('Distribuição da Temperatura no Momento dos Acidentes')
plt.ylabel('Temperatura (C)')

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
df1['Acident_Duration'] = df['End_Time'] - df['Start_Time']

print(df1)
print(df1['Temperature(F)'])
print(df1['Temperature(C)'])

