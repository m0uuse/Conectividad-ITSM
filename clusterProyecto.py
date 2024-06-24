import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos con el delimitador correcto
df = pd.read_csv('C:/ccd/dataset_velocidad_internet.csv', delimiter=',', encoding='latin1')

# Mostrar las columnas del DataFrame
print(f"Columnas del dataset: {df.columns}")

# Seleccionar las características para clustering (excluyendo 'Promedio (Mbps)')
features = ['Carga (Mbps)', 'Descarga (Mbps)', 'Area']
X = df[features].dropna()

# Filtrar solo las columnas numéricas para el escalado
numeric_features = ['Carga (Mbps)', 'Descarga (Mbps)']
X_numeric = X[numeric_features]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Determinar el número óptimo de clusters usando el método del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Graficar el método del codo
plt.plot(range(1, 11), wcss)
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

# Seleccionar el número óptimo de clusters (por ejemplo, 4)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Asignar etiquetas descriptivas a los clusters
cluster_labels = {0: 'Buena conexión', 1: 'Conexión regular', 2: 'Conexión muy mala', 3: 'Conexión mala'}
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

# Ordenar el DataFrame por la columna 'Cluster'
df = df.sort_values(by='Cluster')

# Evaluar el cluster con mejor performance basado en las velocidades de carga y descarga
cluster_performance = df.groupby('Cluster')[numeric_features].mean()
best_cluster = cluster_performance[['Carga (Mbps)', 'Descarga (Mbps)']].mean(axis=1).idxmax()

print("Cluster con mejor rendimiento:")
print(f"Cluster {best_cluster} ({cluster_labels[best_cluster]})")
print(cluster_performance.loc[best_cluster])

# Mostrar las áreas asociadas con el cluster de mejor rendimiento
areas_mejor_rendimiento = df[df['Cluster'] == best_cluster]['Area'].unique()
print("Áreas con mejor rendimiento de internet:")
print(areas_mejor_rendimiento)

# Crear un ranking de las áreas basadas en la velocidad de descarga de internet
area_ranking = df.groupby('Area')['Descarga (Mbps)'].mean().sort_values(ascending=False)
print("Ranking de áreas basado en la velocidad de descarga de internet:")
print(area_ranking)

# Graficar los clusters
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for i, feature in enumerate(numeric_features):
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        axes[i].hist(cluster_data[feature], alpha=0.5, label=f'Cluster {cluster} ({cluster_labels[cluster]})')
    axes[i].set_title(f'Histograma de {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frecuencia')
    axes[i].legend()

plt.tight_layout()
plt.show()

# Graficar la distribución de áreas en función de los clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster_Label', data=df, palette='viridis')
plt.title('Distribución de Áreas en Clusters')
plt.xlabel('Cluster')
plt.ylabel('Cantidad de Áreas')
plt.legend(title='Área', loc='upper right')
plt.grid(True)
plt.show()

# Graficar los clusters en función del área
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Descarga (Mbps)', y='Carga (Mbps)', hue='Cluster_Label', data=df, palette='viridis', alpha=0.7)
plt.title('Distribución de Clusters en Áreas')
plt.xlabel('Velocidad de Descarga (Mbps)')
plt.ylabel('Velocidad de Carga (Mbps)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Graficar los promedios de cada cluster en una gráfica de barras
cluster_performance.plot(kind='bar', figsize=(12, 8))
plt.title('Promedio de Velocidades por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Velocidad (Mbps)')
plt.grid(True)
plt.show()

# Mostrar los clusters y sus características promedio
print("Características promedio de cada cluster:")
print(df.groupby('Cluster')[numeric_features].mean())

# Crear una tabla de contingencia para contar las áreas en cada cluster
contingency_table = pd.crosstab(df['Area'], df['Cluster'])

# Determinar el cluster al que pertenece cada área
area_to_cluster = contingency_table.idxmax(axis=1)

# Ordenar el resultado por cluster
area_to_cluster = area_to_cluster.sort_values()

# Asignar etiquetas descriptivas a cada área basado en el conteo
area_to_cluster_labels = area_to_cluster.map(cluster_labels)

# Agregar una nueva columna con el cluster asignado a cada área en el DataFrame original
df['Area_Cluster'] = df['Area'].map(area_to_cluster_labels)

# Guardar el DataFrame actualizado en un archivo CSV
df.to_csv('dataset_velocidad_internet_actualizado.csv', index=False)

print(df.head())

print("Tabla de contingencia de áreas y clusters:")
print(contingency_table)

print("Cluster asignado a cada área basado en el conteo:")
print(area_to_cluster_labels)
