import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)

#Criação de um novo dataframe
df1 = pd.DataFrame(new_df)
#print(df1)

#converter a temperatura de fahrenheit para celsius
df1['Temperature(C)'] = (df['Temperature(F)'] - 32) * 5/9

#converter a distancia de milhas para metros
df1['Distance(M)'] = (df['Distance(mi)'] * 1609.34)

#converter a visibilidade de milhas para metros
df1['Visibility(M)'] = (df['Visibility(mi)'] * 1609.34)

#transformar os dados categóricos em numéricos
df1['Sunrise_Sunset'] = df['Sunrise_Sunset'].map({'Day': 1, 'Night': 0})
df1['Crossing'] = df['Crossing'].map({True: 1, False: 0})
df1['Traffic_Signal'] = df['Traffic_Signal'].map({True: 1, False: 0})
df1['Amenity'] = df['Amenity'].map({True: 1, False: 0})
df1['Bump'] = df['Bump'].map({True: 1, False: 0})
df1['Give_Way'] = df['Give_Way'].map({True: 1, False: 0})
df1['Junction'] = df['Junction'].map({True: 1, False: 0})
df1['No_Exit'] = df['No_Exit'].map({True: 1, False: 0})
df1['Railway'] = df['Railway'].map({True: 1, False: 0})
df1['Roundabout'] = df['Roundabout'].map({True: 1, False: 0})
df1['Station'] = df['Station'].map({True: 1, False: 0})
df1['Stop'] = df['Stop'].map({True: 1, False: 0})
df1['Traffic_Calming'] = df['Traffic_Calming'].map({True: 1, False: 0})

#converter os dados de tempo para datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])
#calcular a diferenca entre o tempo de inicio e fim do acidente
df1['Acident_Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds().astype(int)

def describe():
    #Ordena a severidade por quartis (25%, 50%(mediana), 75%)
    #describSever = df.describe()
    #print(describSever)

    #print(df1.describe())

    #print(df1.size)

def grafico_acidentes_timezone():
    #Acidentes nas diferentes timezones
    timezone_counts = df1['Timezone'].value_counts().head(4)
    timezone_counts.plot(kind="bar")
    plt.title("Numero de acidentes por time zone")
    plt.xlabel("Timezone")
    plt.ylabel("Count") 

    plt.show()

def grafico_nAcidentes_severidade():
    severity_humidity = df1.groupby('Severity')['Humidity(%)'].mean()
    severity_humidity.plot(kind="bar")
    plt.title("Severidade Por media de Humidade")
    plt.xlabel("Severidade")
    plt.ylabel("Humidade")
    severity_humidity.corr()
    plt.show()

def garfico_Weather_Acidentes():
    weather = df1['Weather_Condition'].value_counts().head(7)
    weather.plot(kind="bar")
    plt.title("Estados de tempo em que ocorrem mais acidentes")
    plt.xlabel("Tempo")
    plt.ylabel("Acidentes") 

    plt.show()

def grafico_acidentes_estado():
    countsAciSta = df1["State"].value_counts()
    countsAciSta.plot(kind="bar")
    plt.title("Acidentes por estado")
    plt.xlabel("State")
    plt.ylabel("Count") 

    plt.show()

def correlacoes():  
    num_df = df1.select_dtypes(include = 'number')
    df2 = num_df.drop(columns=['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Number'])
    sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.show()

def dispersao():
    #cria grafico de dispersão entre a severidade,temperatura em Celcius, visibilidade e Humidade
    sns.pairplot(df1[['Severity', 'Humidity(%)', 'Temperature(C)', 'Visibility(M)']])
    plt.show()


def histograma_acidentes_por_mes():
    #HISTOGRAMA E CONTAGEM DO NºACIDENTES POR MÊS
    df1['Start_Time'] = pd.to_datetime(df['Start_Time']) # converte a coluna de data em um objeto de série de data
    df1['Month_Year'] = df1['Start_Time'].dt.year.astype(str) + '/' + df1['Start_Time'].dt.month.astype(str)
    x = df1['Month_Year'].value_counts() #Contagem do nºde acidentes por mês
    x = x.sort_index()
    x.plot(kind="bar")

    #Adiciona títulos e rótulos aos eixos
    plt.title('Distribuição dos Acidentes de Carro ao Longo dos Meses')
    plt.xlabel('Mês')
    plt.ylabel('Número de Acidentes')

    plt.show()

def normalizacao():
    plt.plot(df1['Severity'])
    plt.xlim(0, 100)
    plt.figure()
    #fit scaler on data
    normMinMax = MinMaxScaler()
    # transform data
    norm = normMinMax.fit_transform(df[['Severity']].values)
    print(norm)
    plt.plot(norm)
    plt.xlim(0, 100)
    plt.show()

def boxPlots():
    #SEVERIDADE-------------------------------------------------------
    # cria um boxplot da coluna "Severity"
    plt.boxplot(df1['Severity'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição da Severidade dos Acidentes')
    plt.ylabel('Severidade')

    # exibe o boxplot
    plt.show()

    #Wind_Chill-------------------------------------------------------
    # cria um boxplot da coluna "Wind_Chill"
    plt.boxplot(df1['Wind_Chill(F)'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição do wind chill')
    plt.ylabel('Wind_Chill')

    # exibe o boxplot
    plt.show()

    #Humidity-------------------------------------------------------
    # cria um boxplot da coluna "humidity"
    plt.boxplot(df1['Humidity(%)'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição da humidade')
    plt.ylabel('Humidade')

    # exibe o boxplot
    plt.show()

    #Pressão-------------------------------------------------------
    # cria um boxplot da coluna "Pressao"
    plt.boxplot(df1['Pressure(in)'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição da pressao atmosferica')
    plt.ylabel('Pressão')

    # exibe o boxplot
    plt.show()

    #Visibility-------------------------------------------------------
    # cria um boxplot da coluna "Visibility"
    plt.boxplot(df1['Visibility(mi)'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição da visibilidade')
    plt.ylabel('Visibilidade')

    # exibe o boxplot
    plt.show()

    #Wind_Speed-------------------------------------------------------
    # cria um boxplot da coluna "Wind_Speed"
    plt.boxplot(df1['Wind_Speed(mph)'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição da velocidade do vento')
    plt.ylabel('wind_Speed')

    # exibe o boxplot
    plt.show()

    #Precipitação-------------------------------------------------------
    # cria um boxplot da coluna "Precipitação"
    plt.boxplot(df1['Precipitation(in)'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição da precipitação')
    plt.ylabel('Precipitação')

    # exibe o boxplot
    plt.show()

    #Duração-----------------------------------------------------
    # cria um boxplot da coluna "Duration"
    plt.boxplot(df1['Acident_Duration'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição da Duração dos Acidentes')
    plt.ylabel('Duração')

    # exibe o boxplot
    plt.show()

    #Distancia-----------------------------------------------------
    # cria um boxplot da coluna "Distancia"
    plt.boxplot(df1['Distance(M)'])

    # adiciona títulos e rótulos aos eixos
    plt.title('Distribuição da Distancia dos Acidentes')
    plt.ylabel('Distância')

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

def transformacao_linearDistancia():
    plt.plot(df1['Distance(mi)'])
    plt.title('Sem transformação linear(mi)')
    plt.xlim(0, 100)
    plt.figure()
    plt.plot(df1['Distance(M)'])
    plt.title('Com transformação linear(M)')
    plt.xlim(0, 100)
    plt.show()

def transformacao_linearTemperatura():
        plt.plot(df1['Temperature(F)'])
        plt.title('Sem transformação linear(F)')
        plt.xlim(0, 100)
        plt.figure()
        plt.plot(df1['Temperature(C)'])
        plt.title('Com transformação linear(C)')
        plt.xlim(0, 100)
        plt.show()

#grafico_acidentes_estado()
#histograma_acidentes_por_mes()
#normalizacao()
#transformacao_linearDistancia()
#boxPlots()
#describe()
#correlacoes()
#dispersao()
#grafico_acidentes_timezone()
#grafico_severidade_humidade()
#garfico_Weather_Acidentes()
#transformacao_linearTemperatura()

