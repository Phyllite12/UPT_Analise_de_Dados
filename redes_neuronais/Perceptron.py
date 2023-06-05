import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# Carregar o conjunto de dados
df = pd.read_csv('US_Accidents_Dec21_updated.csv')

# Preencher os valores NaN com zero no DataFrame original
df.fillna(0, inplace=True)

# Criar uma nova cópia do DataFrame
df1 = pd.DataFrame(df)

# Selecionar as colunas relevantes para o modelo Perceptron
selected_columns = ['Visibility(mi)', 'Precipitation(in)']  # Substitua pelos nomes das colunas relevantes do seu dataset

# Filtrar o dataset usando apenas as colunas relevantes
X_features = df[selected_columns].values
y_labels = df['Severity'].values  # Substitua 'Severity' pelo nome da coluna alvo do seu dataset

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.5, random_state=0)

# Treinar o modelo Perceptron
clf = Perceptron()
clf.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = clf.score(X_test, y_test)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_mat)
