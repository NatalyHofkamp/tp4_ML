import numpy as np
import matplotlib.pyplot as plt

class K_means: 
    def __init__(self, k=2, tolerance=0.001, max_iter=50):
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


def Kmeans_plot_clusters(clusters_data,k):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap('tab20').colors
    clusters, centroids = clusters_data['clusters'], clusters_data['centroids']
    for cluster in clusters:
        cluster_data = np.array(clusters[cluster])
        color = colors[cluster % len(colors)]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color,  alpha=0.6)
        plt.scatter(centroids[cluster][0], centroids[cluster][1], color='k', marker='x', s=100, linewidths=3)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'K={k}', fontsize=12)
        
def Kmeans_inertia(K_values, inertia_values):
    plt.plot(K_values, inertia_values, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.show()
