import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Carregar o conjunto de dados
df = pd.read_csv('US_Accidents_Dec21_updated.csv')

# Preencher os valores NaN com zero no DataFrame original
df.fillna(0, inplace=True)

# Criar uma nova cópia do DataFrame
df1 = pd.DataFrame(df)

# Transformar dados categóricos em numéricos
df1['Sunrise_Sunset'] = df1['Sunrise_Sunset'].map({'Day': 1, 'Night': 0})
df1['Crossing'] = df1['Crossing'].map({True: 1, False: 0})
df1['Traffic_Signal'] = df1['Traffic_Signal'].map({True: 1, False: 0})
df1['Amenity'] = df1['Amenity'].map({True: 1, False: 0})
df1['Bump'] = df1['Bump'].map({True: 1, False: 0})
df1['Give_Way'] = df1['Give_Way'].map({True: 1, False: 0})
df1['Junction'] = df1['Junction'].map({True: 1, False: 0})
df1['No_Exit'] = df1['No_Exit'].map({True: 1, False: 0})
df1['Railway'] = df1['Railway'].map({True: 1, False: 0})
df1['Roundabout'] = df1['Roundabout'].map({True: 1, False: 0})
df1['Station'] = df1['Station'].map({True: 1, False: 0})
df1['Stop'] = df1['Stop'].map({True: 1, False: 0})
df1['Traffic_Calming'] = df1['Traffic_Calming'].map({True: 1, False: 0})

# Converter temperatura de Fahrenheit para Celsius
df1['Temperature(C)'] = (df1['Temperature(F)'] - 32) * 5/9

# Converter distância de milhas para metros
df1['Distance(M)'] = df1['Distance(mi)'] * 1609.34

# Converter dados de tempo para formato datetime
df1['Start_Time'] = pd.to_datetime(df1['Start_Time'])
df1['End_Time'] = pd.to_datetime(df1['End_Time'])
# Calcular a diferença entre o tempo de início e término do acidente
df1['Accident_Duration'] = (df1['End_Time'] - df1['Start_Time']).dt.total_seconds().astype(int)

# Definir os dados de entrada e saída
dfInput = ['Distance(M)', 'Temperature(C)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Sunrise_Sunset']
dfOutput = df1['Severity']

# Preencher valores NaN nas colunas de entrada especificadas com zero
df1[dfInput] = df1[dfInput].fillna(0)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df1[dfInput], dfOutput, test_size=0.3, random_state=109)

# Criar um classificador AdaBoost com uma árvore de decisão como estimador base
adaboost = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(criterion='gini', max_depth=2),
                              n_estimators=50,  # Número de estimadores (árvores)
                              learning_rate=1.0)  # Taxa de aprendizado

# Treinar o modelo
adaboost.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = adaboost.predict(X_test)

# Calcular a precisão do modelo
accuracy = adaboost.score(X_test, y_test)
print("Precisão:", accuracy)

