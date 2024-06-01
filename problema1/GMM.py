import numpy as np
import matplotlib.pyplot as plt
from K_means import K_means 
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, max_iter=50):
        self.k = k
        self.max_iter = int(max_iter)
    
    def initialize(self, X):
        """
        Initialize GMM parameters.
        """
        self.n, self.m = X.shape
        # Encontrar los clusters y centroides usando K-means
        kmeans = K_means(self.k,0.001,20)
        self.clusters, self.centroids = kmeans.set_clusters(X)

        self.mu = self.centroids
        self.pi = np.array([len(self.clusters[i]) for i in range(self.k)]) / len(X)
        self.sigma = [np.cov(np.array(self.clusters[i]).T) if len(self.clusters[i]) > 0 else np.eye(self.m) for i in range(self.k)]
        
    def E_step(self, X):
        """
        Perform the expectation step (E-step) of the EM algorithm.
        """
        N = len(X)
        responsibilities = np.zeros((N, self.k))
        for point in range(N):
            for cluster in range(self.k):
                if self.sigma[cluster].ndim == 0:  # Check if sigma is correctly initialized
                    raise ValueError(f"Sigma for cluster {cluster} is not properly initialized.")
                responsibilities[point, cluster] = self.pi[cluster] * multivariate_normal.pdf(X[point], self.mu[cluster], self.sigma[cluster])
            if np.sum(responsibilities[point, :]) != 0:
                responsibilities[point, :] /= np.sum(responsibilities[point, :])
            else:
                responsibilities[point, :] = 1 / self.k
        return responsibilities

    def M_step(self, X):
        """
        Perform the maximization step (M-step) of the EM algorithm.
        """
        new_mu = np.zeros((self.k, self.m))
        new_sigma = np.zeros((self.k, self.m, self.m))
        new_pi = np.zeros(self.k)
        for k in range(self.k):
            Nk = np.sum(self.responsibilities[:, k], axis=0)
            new_mu[k] = np.sum(self.responsibilities[:, k][:, np.newaxis] * X, axis=0) / Nk
            diff = X - new_mu[k]
            new_sigma[k] = np.dot(self.responsibilities[:, k] * diff.T, diff) / Nk
            new_pi[k] = Nk / len(X)
        return new_mu, new_sigma, new_pi

    def check_loglikelihood(self, X, log_likelihoods, tol):
        """
        Check convergence based on log-likelihood.
        """
        log_likelihood = np.sum(np.log(np.sum([self.pi[k] * multivariate_normal.pdf(X, self.mu[k], self.sigma[k]) for k in range(self.k)], axis=0)))
        log_likelihoods.append(log_likelihood)
        if len(log_likelihoods) > 1 and abs(log_likelihood - log_likelihoods[-2]) < tol:
            return True
        return False

    def Train(self, X, max_iters=100, tol=1e-3):
        """
        Train the GMM model using the EM algorithm.
        """
        self.initialize(X)
        log_likelihoods = []
        for _ in range(max_iters):
            self.responsibilities = self.E_step(X)
            self.mu, self.sigma, self.pi = self.M_step(X)
            if self.check_loglikelihood(X, log_likelihoods, tol):
                break
        return log_likelihoods
    
    def set_clusters(self, X):
        """
        Assign data points to clusters based on responsibilities.
        """
        cluster_assignments = np.argmax(self.responsibilities, axis=1)
        clusters = {i: [] for i in range(self.k)}
        for idx, cluster in enumerate(cluster_assignments):
            clusters[cluster].append(X[idx])
        return clusters
        
def plot_clusters_and_gaussians(X, clusters, mu, sigma, ax):
    colors = plt.get_cmap('tab20').colors
    for cluster_idx, points in clusters.items():
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], color=colors[cluster_idx % len(colors)], label=f'Cluster {cluster_idx}')
    
    x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    pos = np.dstack((xx, yy))
    
    for k in range(len(mu)):
        rv = multivariate_normal(mu[k], sigma[k])
        ax.contour(xx, yy, rv.pdf(pos), levels=10, colors=[colors[k % len(colors)]], alpha=0.5)

    ax.legend()

def plot_models(model, data, k):
    fig, ax = plt.subplots(figsize=(10, 8))
    clusters, mu, sigma = model
    plot_clusters_and_gaussians(data, clusters, mu, sigma, ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title(f'Gaussian Mixture Model with {k} Clusters')
    plt.show()

    
def calculate_inertia(clusters):
    inertia = 0
    for cluster_idx, points in clusters.items():
        center = np.mean(points, axis=0)
        inertia += np.sum(np.linalg.norm(points - center)** 2)
    return inertia

def GMM_inertia(inertias):
    k_values, iner = zip(*inertias.items())
    plt.plot(k_values, iner, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of Clusters')
    plt.show()