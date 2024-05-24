import numpy as np
import matplotlib.pyplot as plt
from dataset import *


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit_predict(self, X):
        self.labels_ = np.zeros(X.shape[0], dtype=int)  # Todos los puntos son inicializados como ruido (-1)
        cluster_id = 0

        for i, point in enumerate(X):
            if self.labels_[i] != 0:  # Ya fue asignado a un cluster
                continue
            
            neighbors = self._get_neighbors(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Marcar como ruido
            else:
                cluster_id += 1
                self._expand_cluster(X, i, neighbors, cluster_id)

        return self.labels_

    def _get_neighbors(self, X, i):
        return [j for j, point in enumerate(X) if np.linalg.norm(point - X[i]) <= self.eps]

    def _expand_cluster(self, X, i, neighbors, cluster_id):
        self.labels_[i] = cluster_id

        j = 0
        while j < len(neighbors):
            neighbor_index = neighbors[j]
            if self.labels_[neighbor_index] == -1:  # Si el vecino es ruido, lo marcamos como borde
                self.labels_[neighbor_index] = cluster_id
            elif self.labels_[neighbor_index] == 0:  # Si no ha sido asignado a ningún cluster
                self.labels_[neighbor_index] = cluster_id
                new_neighbors = self._get_neighbors(X, neighbor_index)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            j += 1
def plot_clusters(X, labels):
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'gray'  # Ruido en gris
        class_member_mask = (labels == label)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], color=color, s=30, label=f'Cluster {label}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('DBSCAN Clustering')
    plt.legend()
    plt.show()

def main():
    data = get_data('problema1/clustering.csv')
    
    eps_values = [0.1, 0.5, 1.0]  # Variar el radio de la vecindad
    min_samples_values = [5, 10, 20]  # Variar el mínimo número de puntos en una zona densa

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            plot_clusters(data, labels)

if __name__ == "__main__":
    main()
