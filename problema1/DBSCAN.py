import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, K=5):
        self.eps = eps
        self.K = K
        self.labels_ = None

    def get_neighbors(self, X, point_idx):
        """
        Get the indices of the neighboring points within distance eps of a given point.
        """
        neighbors = []
        for i in range(len(X)):
            if np.linalg.norm(X[i] - X[point_idx]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def fit_predict(self, X):
        """
        Fit the DBSCAN clustering algorithm to the data and return cluster assignments.
        """
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        cluster_id = 0

        for i in range(len(X)):
            if self.labels_[i] != 0:  # Already classified
                continue
            
            neighbors = self.get_neighbors(X, i)
            
            if len(neighbors) < self.K:
                self.labels_[i] = -1  # Mark as noise
            else:
                cluster_id += 1
                self.expand_cluster(X, i, neighbors, cluster_id)
        
        clusters = {}
        for idx, label in enumerate(self.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(X[idx])
        
        for label in clusters:
            clusters[label] = np.array(clusters[label])
        
        return clusters

    def expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """
        Expand the cluster by adding neighboring points to the current cluster.
        """
        self.labels_[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            
            if self.labels_[neighbor_idx] == 0:
                self.labels_[neighbor_idx] = cluster_id
                new_neighbors = self.get_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.K:
                    neighbors += new_neighbors
            i += 1

def plot_clusters(clusters, e, k):
    """
    Plot clusters obtained from DBSCAN algorithm.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap('tab20').colors

    for cluster_id, cluster_data in clusters.items():
        color = 'k' if cluster_id == -1 else colors[cluster_id % len(colors)]
        label = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color, label=label, alpha=0.6)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'DBSCAN with $\\epsilon$ = {e} and K: {k}', fontsize=12)
    plt.legend()
    plt.show()

def DBSCAN_inertia(inertias):
    """
    Plot inertia values for different combinations of epsilon and K.
    """
    eps_k_pairs, iner = zip(*inertias.items())
    eps_values, k_values = zip(*eps_k_pairs)
    plt.plot(range(len(iner)), iner, marker='o')
    plt.xticks(range(len(iner)), [f'eps={e}, K={k}' for e, k in zip(eps_values, k_values)], rotation=45)
    plt.xlabel(f'Parameter combination ($\\epsilon$ , K)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Parameter Combinations')
    plt.tight_layout()
    plt.show()
