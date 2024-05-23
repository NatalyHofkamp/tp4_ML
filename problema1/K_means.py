import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_data(filename):
    data = pd.read_csv(filename)
    return data.iloc[:, 1:].values  # Convertir a numpy array

class K_means: 
    def __init__(self, k=2, tolerance=0.001, max_iter=500):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance
        self.centroids = {}
        self.clusters = {}

    def euclidean_distance(self, x_1, x_2):
        return np.linalg.norm(x_1 - x_2)

    def choose_cluster(self, x_new):
        distances = [self.euclidean_distance(x_new, self.centroids[centroid]) for centroid in self.centroids]
        return distances.index(min(distances))

    def inertia(self, data):
        total_inertia = 0
        for cluster in self.clusters:
            for x in self.clusters[cluster]:
                total_inertia += self.euclidean_distance(x, self.centroids[cluster]) ** 2
        return total_inertia

    def set_clusters(self, data):
        self.clusters = {i: [] for i in range(self.k)}
        indices = np.random.choice(len(data), self.k, replace=False)
        for i in range(self.k):
            self.centroids[i] = data[indices[i]]

        for i in range(self.max_iterations):
            for x in data:
                x_cluster = self.choose_cluster(x)
                self.clusters[x_cluster].append(x)

            prev_centroids = dict(self.centroids)

            for cluster in self.clusters:
                if len(self.clusters[cluster]) > 0:
                    self.centroids[cluster] = np.mean(self.clusters[cluster], axis=0)

            is_converged = True
            for centroid in self.centroids:
                if np.sum((self.centroids[centroid] - prev_centroids[centroid]) / prev_centroids[centroid] * 100.0) > self.tolerance:
                    is_converged = False

            if is_converged:
                break
        return self.clusters, self.centroids

def plot_clusters(all_clusters):
    # Lista de colores ampliada
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    plt.figure(figsize=(12, 10))
    
    for k, clusters_data in all_clusters.items():
        clusters, centroids = clusters_data['clusters'], clusters_data['centroids']
        for cluster in clusters:
            cluster_data = np.array(clusters[cluster])
            color = colors[cluster % len(colors)]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color, label=f'Cluster {cluster} for K={k}', alpha=0.6)
            plt.scatter(centroids[cluster][0], centroids[cluster][1], color='k', marker='x', s=100, linewidths=3)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title(f'Cluster Assignments and Centroids for K={k}')
    plt.show()

def best_K(K_values, inertia_values):
    plt.plot(K_values, inertia_values, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.show()

def main():
    data = get_data('problema1/clustering.csv')
    tolerance = 0.001
    max_iter = 20
    inertia_values = []
    K_values = range(2,10)
    models = {}

    for k in K_values:
        model = K_means(k, tolerance, max_iter)
        clusters, centroids = model.set_clusters(data)
        inertia = model.inertia(data)
        models[k] = {'clusters': clusters, 'centroids': centroids, 'inertia': inertia}
        inertia_values.append(inertia)

    best_K(K_values, inertia_values)

    # Graficar clusters para cada K
    for k in K_values:
        plot_clusters(k, models[k]['clusters'], models[k]['centroids'], data)

if __name__ == '__main__':
    main()
