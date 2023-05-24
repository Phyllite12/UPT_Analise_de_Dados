import pandas as pd
from sklearn.cluster import KMeans
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
X_features = df[selected_columns].values

# Calcular a soma dos quadrados das distâncias para diferentes números de clusters
wcss = []  # Soma dos quadrados das distâncias
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_features)
    wcss.append(kmeans.inertia_)

# Plotar o gráfico do método do cotovelo
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
