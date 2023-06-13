import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
df = pd.read_csv('US_Accidents_Dec21_updated.csv')

# Preencher os valores NaN com zero no DataFrame original
df.fillna(0, inplace=True)

# Criar uma nova cópia do DataFrame
df1 = pd.DataFrame(df)

# Selecionar as colunas relevantes para o clustering
selected_columns = ['Severity', 'Temperature(F)', 'Visibility(mi)']

# Filtrar o dataset usando apenas as colunas relevantes
X_features = df1[selected_columns].values

# Amostragem aleatória dos dados
sample_size = 10000  # Defina o tamanho da amostra desejada
random_indices = np.random.choice(len(X_features), size=sample_size, replace=False)
X_features_sample = X_features[random_indices]

# Criar o modelo de clustering
model = AgglomerativeClustering(linkage="ward", n_clusters=3)

# Realizar o clustering
predicted_labels = model.fit_predict(X_features_sample)

# Construir a matriz de linkage
linkage_matrix = linkage(X_features_sample, 'ward')

# Plotar o dendrograma
plot = plt.figure(figsize=(14, 7))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20, color_threshold=0)
plt.title('Hierarchical Clustering Dendrogram (linkage=ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.show()
