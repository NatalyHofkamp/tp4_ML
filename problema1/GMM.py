import numpy as np
import matplotlib.pyplot as plt 
from K_means import K_means  # Asegúrate de que K_means esté correctamente implementado e importado
from dataset import get_data  # Asegúrate de que get_data esté correctamente implementado e importado

class GMM:
    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape
        # Inicialización usando K-means
        kmeans = K_means(self.k)
        clusters, centroids = kmeans.set_clusters(X)

        self.mu = centroids
        self.phi = np.array([np.sum(clusters == i) for i in range(self.k)]) / self.n
        self.sigma = [np.cov(X[clusters == i].T) for i in range(self.k)]
        self.weights = np.full((self.n, self.k), fill_value=1/self.k)

    def E_step(self, X):
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)

    def M_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, aweights=(weight / total_weight).flatten(), bias=True)

    def multivariate_gaussian(self, X, mean, cov):
        n = X.shape[1]
        diff = X - mean
        exp_term = np.exp(-0.5 * np.einsum('ij,ij->i', np.dot(diff, np.linalg.inv(cov)), diff))
        norm_term = np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))
        return exp_term / norm_term

    def predict_proba(self, X):
        likelihood = np.zeros((self.n, self.k))
        for i in range(self.k):
            likelihood[:, i] = self.multivariate_gaussian(X, self.mu[i], self.sigma[i])

        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)

    def inertia(self, X):
        inertia = 0
        for i in range(self.n):
            distances = np.linalg.norm(X[i] - self.mu, axis=1)
            nearest_centroid_index = np.argmin(distances)
            inertia += distances[nearest_centroid_index] ** 2
        return inertia

def plot_elbow(inertia_values, K_values):
    plt.plot(K_values, inertia_values, marker='o')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Inercia')
    plt.title('Método del codo para K óptimo')
    plt.show()

def main():
    max_K = 15  # Definir el máximo valor de K a probar
    max_iter = 50
    data = get_data('problema1/clustering.csv')
    model = GMM(2, max_iter)  # Inicializar el modelo con K=1

    inertia_values = []

    for k in range(1, max_K + 1):
        model.k = k
        model.initialize(data)

        for _ in range(max_iter):
            model.E_step(data)
            model.M_step(data)

        inertia = model.inertia(data)
        inertia_values.append(inertia)

    K_values = range(1, max_K + 1)  # Número de clusters es el rango de K
    plot_elbow(inertia_values, K_values)


if __name__ == "__main__":
    main()
