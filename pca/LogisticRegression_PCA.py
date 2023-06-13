from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def predict_severity(dfInput_values):
    df = pd.read_csv('US_Accidents_Dec21_updated.csv')
    df.fillna(0, inplace=True)

    df1 = pd.DataFrame(df)

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

    df1['Temperature(C)'] = (df1['Temperature(F)'] - 32) * 5/9
    df1['Distance(M)'] = df1['Distance(mi)'] * 1609.34

    df1['Start_Time'] = pd.to_datetime(df1['Start_Time'])
    df1['End_Time'] = pd.to_datetime(df1['End_Time'])
    df1['Accident_Duration'] = (df1['End_Time'] - df1['Start_Time']).dt.total_seconds().astype(int)

    dfInput = ['Distance(M)', 'Temperature(C)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
               'Wind_Speed(mph)', 'Precipitation(in)', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
               'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Sunrise_Sunset']
    dfOutput = df1['Severity']

    df1[dfInput] = df1[dfInput].fillna(0)

    X, y = df1[dfInput], dfOutput

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    clf = LogisticRegression(random_state=0, max_iter=100000)
    clf.fit(X_train_pca, y_train)

    X_input = pd.DataFrame([dfInput_values], columns=dfInput)
    severity = cross_val_score(clf, X, y, cv=5)

    return severity

