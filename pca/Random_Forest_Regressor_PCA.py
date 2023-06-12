import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregar o conjunto de dados
df = pd.read_csv('C:\\Users\\marco\\Documents\\VsCode\python\\Traballho\\Analise_de_Dados\\US_Accidents_Dec21_updated.csv')

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

# Standardize the input data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to the standardized data
pca = PCA(n_components=10)  # Set the desired number of components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Criar o regressor RandomForestRegressor
rf = RandomForestRegressor(max_depth=10, random_state=0)

# Treinar o modelo usando os conjuntos de treinamento com PCA
rf.fit(X_train_pca, y_train)

# Prever a resposta para o conjunto de teste com PCA
y_pred = rf.predict(X_test_pca)

# Calcular o coeficiente de determinação (R²) e o erro médio absoluto (MAE)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Imprimir os resultados
print("R2 = {:.2f}".format(r2))
print("MAE = {:.2f}".format(mae))
