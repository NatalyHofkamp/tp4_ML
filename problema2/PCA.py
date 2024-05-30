import numpy as np
import matplotlib.pyplot as plt
from dataset import get_data

class PCA:
    def __init__(self, data):
        self.data = data
        self.mean = np.mean(data, axis=0)
        self.centered_data = data - self.mean
        self.eigen_values = []
        self.eigen_vectors = []

    def get_covariance(self):
        # Covariance matrix
        return np.cov(self.centered_data, rowvar=False)
    
    def get_eigen(self, cov_mat):
        # Eigenvalues and eigenvectors of the covariance matrix
        eigen_val, eigen_vec = np.linalg.eig(cov_mat)
        self.eigen_values = eigen_val
        self.eigen_vectors = eigen_vec
    
    def sort_eigen(self):
        # Sort eigenvalues and eigenvectors
        sort_idx = np.argsort(np.abs(self.eigen_values))[::-1]
        self.eigen_values = self.eigen_values[sort_idx]
        self.eigen_vectors = self.eigen_vectors[:, sort_idx]

    def transform(self, n_components):
        # Project data onto the top n_components principal components
        return np.dot(self.centered_data, self.eigen_vectors[:, :n_components])
    
    def inverse_transform(self, transformed_data, n_components):
        # Reconstruct data from the top n_components principal components
        return np.dot(transformed_data, self.eigen_vectors[:, :n_components].T) + self.mean

def plot_img(image, title):
    plt.figure()
    plt.imshow(np.array(image).reshape(28, 28), clim=(-1, 1.0), cmap='gray_r')
    plt.title(title)
    plt.show()

def main():
    # Load data
    data = get_data("problema2/MNIST_dataset.csv")
    
    # Initialize PCA
    pca = PCA(data)
    
    # Compute covariance matrix and eigenvalues/eigenvectors
    cov_mat = pca.get_covariance()
    pca.get_eigen(cov_mat)
    pca.sort_eigen()

    # Calculate reconstruction error for different number of components
    mse_errors = []
    num_components = range(1, 101)  # Change the range according to the dataset dimensions

    for n in num_components:
        transformed_data = pca.transform(n)
        reconstructed_data = pca.inverse_transform(transformed_data, n)
        mse = np.mean((data - reconstructed_data) ** 2)
        mse_errors.append(mse)
    
    # Plot MSE vs. number of components
    plt.plot(num_components, mse_errors, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Number of Principal Components')
    plt.show()

    # Select an optimal number of components (e.g., based on the elbow method or a desired MSE threshold)
    optimal_components = 20  # Example: adjust based on the plot

    # Plot original and reconstructed images for the first 10 samples
    transformed_data = pca.transform(optimal_components)
    reconstructed_data = pca.inverse_transform(transformed_data, optimal_components)

    for i in range(10):
        plot_img(data[i], f'Original Image {i + 1}')
        plot_img(reconstructed_data[i], f'Reconstructed Image {i + 1} ({optimal_components} components)')

if __name__ == '__main__':
    main()
