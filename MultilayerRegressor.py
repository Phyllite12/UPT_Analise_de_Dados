from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Carregar o conjunto de dados
df = pd.read_csv('US_Accidents_Dec21_updated.csv')

# Preencher os valores NaN com zero no DataFrame original
df.fillna(0, inplace=True)

# Criar uma nova cópia do DataFrame
df1 = pd.DataFrame(df)

# Selecionar as colunas relevantes para as features e a target column
selected_features = ['Visibility(mi)', 'Precipitation(in)']
target_column = 'Severity'

# Filtrar o dataset usando apenas as colunas relevantes
X_features = df[selected_features].values
y_target = df[target_column].values

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3)

# Criar o regressor MLP
mlp_regressor = MLPRegressor(max_iter=500)

# Treinar o modelo
mlp_regressor.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = mlp_regressor.predict(X_test)

# Avaliar o desempenho do modelo
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score: {:.2f}".format(r2))
print("Mean Absolute Error: {:.2f}".format(mae))
