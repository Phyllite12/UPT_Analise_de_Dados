from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
df = pd.read_csv('US_Accidents_Dec21_updated.csv')

# Preencher os valores NaN com zero no DataFrame original
df.fillna(0, inplace=True)

# Selecionar as colunas relevantes para o clustering
selected_columns = ['Severity', 'Temperature(F)', 'Visibility(mi)']  # Substitua pelos nomes das colunas relevantes do seu dataset

# Filtrar o dataset usando apenas as colunas relevantes
X_features = df[selected_columns].values

# Definir o número de clusters desejado
n_clusters = 3

# Criar o modelo K-means
model = KMeans(n_clusters=3, n_init=10)


# Realizar o clustering
labels = model.fit_predict(X_features)

# Calcular a pontuação de silhueta
silhouette_avg = silhouette_score(X_features, labels)

print("Silhouette Score:", silhouette_avg)

# Visualizar os clusters
plt.scatter(X_features[:, 0], X_features[:, 1], c=labels)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=400, c='black', marker="x", label='Centroids')
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.title("Clustering Results")
plt.legend()
plt.show()
