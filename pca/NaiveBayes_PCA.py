import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('US_Accidents_Dec21_updated.csv')

# Fill NaN values with zero in the original DataFrame
df.fillna(0, inplace=True)

# Create a new DataFrame
df1 = pd.DataFrame(df)

# Transform categorical data into numerical
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

# Convert temperature from Fahrenheit to Celsius
df1['Temperature(C)'] = (df1['Temperature(F)'] - 32) * 5/9

# Convert distance from miles to meters
df1['Distance(M)'] = df1['Distance(mi)'] * 1609.34

# Convert time data to datetime
df1['Start_Time'] = pd.to_datetime(df1['Start_Time'])
df1['End_Time'] = pd.to_datetime(df1['End_Time'])
# Calculate the difference between start and end time of the accident
df1['Accident_Duration'] = (df1['End_Time'] - df1['Start_Time']).dt.total_seconds().astype(int)

# Define input and output data
dfInput = ['Distance(M)', 'Temperature(C)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Sunrise_Sunset']
dfOutput = df1['Severity']

# Fill NaN values with 0 in the specified input features
df1[dfInput] = df1[dfInput].fillna(0)

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df1[dfInput])

# Apply PCA to the standardized features
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Select the desired number of principal components (e.g., 2)
n_components = 2
X_selected = X_pca[:, :n_components]

# Dividing the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, dfOutput, test_size=0.5, random_state=0)

# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the model
y_pred = gnb.fit(X_train, y_train).predict(X_test)

# Display the number of misclassified points and the classification report
print("Number of misclassified points out of a total of %d points: %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(classification_report(y_test, y_pred))

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gnb.classes_, yticklabels=gnb.classes_)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()
