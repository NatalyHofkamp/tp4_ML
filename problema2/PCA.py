import numpy as np
import matplotlib.pyplot as plt


def plot_images(images, title):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Imagen {i+1}')
    plt.show()

class PCA:
    def __init__(self, data):
        self.data = data
        self.eigen_values = []
        self.eigen_vectors = []
        self.transformed_data = None
        self.reconstructed_data = None

        self.get_covariance()
        self.get_eigen()
        self.sort_eigen()
       
    def get_covariance(self):
        """
        Compute the covariance matrix of the input data.
        """
        self.cov_mat = np.cov(self.data, rowvar=False)
    
    def get_eigen(self):
        """
        Compute the eigenvalues and eigenvectors of the covariance matrix.
        """
        eigen_val, eigen_vec = np.linalg.eig(self.cov_mat)
        self.eigen_values = eigen_val
        self.eigen_vectors = eigen_vec
    
    def sort_eigen(self):
        """
        Sort eigenvalues and eigenvectors in descending order based on eigenvalues.
        """
        eig_pairs = [(np.abs(self.eigen_values[i]), self.eigen_vectors[:,i]) for i in range(len(self.eigen_values))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        self.eigen_values = np.array([x[0] for x in eig_pairs])
        self.eigen_vectors = np.array([x[1] for x in eig_pairs]).T

    def transform(self, n_components):
        """
        Transform the input data into a lower-dimensional space using PCA.
        """
        self.transformed_data = np.dot(self.data, self.eigen_vectors[:, :n_components])
        return self.transformed_data
    
    def get_MSE(self, n_components):
        """
        Compute the Mean Squared Error (MSE) between original and reconstructed data.
        """
        self.inverse_transform(n_components)
        mse = np.mean((self.data - self.reconstructed_data) ** 2)
        return mse

    def inverse_transform(self, n_components):
        """
        Reconstruct the original data from the transformed data.
        """
        self.reconstructed_data = np.dot(self.transformed_data[:, :n_components], self.eigen_vectors[:, :n_components].T)
        return self.reconstructed_data

def plot_MSE(num_components, mse_errors):
    """
    Plot the Mean Squared Error (MSE) versus the number of principal components.
    """
    plt.plot(num_components, mse_errors, marker='o', color='purple', label='MSE')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Error Cuadrático Medio')
    plt.title('ECM vs. Número de Componentes Principales')
    plt.grid(True)
    plt.show()