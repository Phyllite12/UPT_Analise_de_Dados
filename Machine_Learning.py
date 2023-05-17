import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('US_Accidents_Dec21_updated.csv')
new_df = df.fillna(0)

#Criação de um novo dataframe
df1 = pd.DataFrame(new_df)

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

#converter a temperatura de fahrenheit para celsius
df1['Temperature(C)'] = (df['Temperature(F)'] - 32) * 5/9

#converter a distancia de milhas para metros
df1['Distance(M)'] = (df['Distance(mi)'] * 1609.34)

#converter os dados de tempo para datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])
#calcular a diferenca entre o tempo de inicio e fim do acidente
df1['Acident_Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds().astype(int)

dfInput = df1.drop(columns = ['Severity', 'Start_Time', 'End_Time', 'Temperature(F)', 'Distance(mi)', 'Star_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Description', 'Street', 'Number', 'Side', 'Zipcode', 'Coutry', 'Airport_Code', 'Weather_Timestamp', 'Wind_Direction', 'Weather_Condition', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'])
dfOutput = df1['Severity']


