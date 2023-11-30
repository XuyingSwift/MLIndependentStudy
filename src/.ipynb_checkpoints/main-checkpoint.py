from generate_fcn_data import generate
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

import numpy as np

def center_data(X):
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    return centered_data, mean

def compute_covariance_matrix(X):
    cov_matrix = np.cov(X, rowvar=False)
    return cov_matrix

def perform_pca(X, num_components):
    centered_data, mean = center_data(X)
    cov_matrix = compute_covariance_matrix(centered_data)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Choose the top 'num_components' eigenvectors
    principal_components = eigenvectors[:, :num_components]
    
    # Project the data onto the principal components
    pca_result = np.dot(centered_data, principal_components)
    
    return pca_result, eigenvalues, principal_components, mean

def filter_eigenvectors(eigenvalues, eigenvectors, threshold):
    # Find the indices of eigenvalues above the threshold
    valid_indices = np.where(eigenvalues > threshold)[0]

    # Select the top eigenvectors based on the threshold
    filtered_eigenvalues = eigenvalues[valid_indices]
    filtered_eigenvectors = eigenvectors[:, valid_indices]

    # 'filtered_eigenvalues' contains the eigenvalues above the threshold
    # 'filtered_eigenvectors' contains the corresponding eigenvectors
    return filtered_eigenvalues, filtered_eigenvectors

def main():
    # generate(10, 10000, "/Users/xuyingwangswift/Documents/MLIndependentStudy/src/data/dimension10/")
    dimension10_ackley = "/Users/xuyingwangswift/Documents/MLIndependentStudy/src/data/dimension10/ackley.csv"

    # Use genfromtxt to read the CSV file into a NumPy array
    dimension10_ackley_data = np.genfromtxt(dimension10_ackley, delimiter=',', skip_header=1)

    #print(dimension10_ackley_data)

    # Example usage:
    # Assuming 'data' is your dataset with each row representing an observation and each column representing a variable
    # Set the number of components you want to keep
    num_components = 2


    # 'pca_result' contains the data projected onto the principal components
    # 'eigenvalues' contains the eigenvalues of the covariance matrix
    # 'principal_components' contains the top 'num_components' eigenvectors
    # 'mean' contains the mean of each variable in the original data
    # Perform PCA
    pca_result, eigenvalues, principal_components, mean = perform_pca(dimension10_ackley_data, num_components)
    print(principal_components)

    # generate(30, 30000, "/Users/xuyingwangswift/Documents/MLIndependentStudy/src/data/dimension30/")
    # generate(50, 50000, "/Users/xuyingwangswift/Documents/MLIndependentStudy/src/data/dimension50/")
    # generate(100, 100000, "/Users/xuyingwangswift/Documents/MLIndependentStudy/src/data/dimension100/")

    # Compute Eigenvalues and Eigenvectors:
    # Compute the eigenvalues and eigenvectors using a method like np.linalg.eigh (for symmetric/Hermitian matrices, like covariance matrices).
    # Filter Based on Threshold:
    # Determine a threshold value for the eigenvalues. Eigenvectors corresponding to eigenvalues below this threshold will be removed.
    # Sort the eigenvalues and corresponding eigenvectors in descending order.
    # Apply Threshold to Eigenvectors:
    # Select the top eigenvectors corresponding to eigenvalues above the threshold.


main()