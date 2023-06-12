from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
df = pd.read_csv('US_Accidents_Dec21_updated.csv')

# Preencher os valores NaN com zero no DataFrame original
df.fillna(0, inplace=True)

# Criar uma nova cópia do DataFrame
df1 = pd.DataFrame(df)

# Selecionar as colunas relevantes para o clustering
selected_columns = ['Severity', 'Temperature(F)', 'Visibility(mi)']  # Substitua pelos nomes das colunas relevantes do seu dataset

# Filtrar o dataset usando apenas as colunas relevantes
X_features = df[selected_columns].values

# Aplicar PCA aos dados
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_features)

# Definir o número de clusters desejado
n_clusters = 3

# Criar o modelo K-means
model = KMeans(n_clusters=n_clusters, n_init=10)

# Realizar o clustering com os dados reduzidos pelo PCA
labels = model.fit_predict(X_pca)

# Calcular a pontuação de silhueta
silhouette_avg = silhouette_score(X_pca, labels)

print("Silhouette Score:", silhouette_avg)

# Visualizar os clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=400, c='black', marker="x", label='Centroids')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title("Clustering Results with PCA")
plt.legend()
plt.show()