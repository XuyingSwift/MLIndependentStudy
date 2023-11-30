import numpy as np
from pso import PSO
from FEA import FEA
from Function import Function
from FactorArchitecture import FactorArchitecture
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
import csv
import os


def load_and_prepare_pca_data(file_path):
    """
    Load and preprocess data.
    """
    df = pd.read_csv(file_path)
    # Standardizing the features
    X_std = StandardScaler().fit_transform(df)
    return X_std

def is_overlapping(eigenvectors, features_above_threshold):
    # Create a fully connected graph
    G = nx.Graph()

    # Add nodes for each eigenvector
    for i in range(len(eigenvectors)):
        G.add_node(i + 1)

    # Add edges based on shared features above the threshold
    for i in range(len(features_above_threshold)):
        for j in range(i + 1, len(features_above_threshold)):
            if any(feature in features_above_threshold[j] for feature in features_above_threshold[i]):
                G.add_edge(i + 1, j + 1)

    # Check if the graph is fully connected
    is_fully_connected = nx.is_connected(G)

    print("The graph is fully connected:", is_fully_connected)

def get_pca_factors(X_std, num_factors):
    """
    Perform PCA and extract factors.
    """
    pca_full = PCA()
    pca_full.fit(X_std)

    # Extracting the eigenvectors
    eigenvectors = pca_full.components_
    return eigenvectors

def get_selected_features(eigenvectors):
    print("Original Eigenvectors:", eigenvectors)

    # Remove the last feature from each eigenvector
    eigenvectors_without_last_feature = [vector[:-1] for vector in eigenvectors]
    print("Eigenvectors without last feature:", eigenvectors_without_last_feature)

    # Define a threshold
    threshold = 0.2  # Adjust as needed
  # Get indices of features in each eigenvector that are above the threshold
    indices_of_features_above_threshold = [
        [index for index, feature in enumerate(vector) if abs(feature) >= threshold] 
        for vector in eigenvectors_without_last_feature
    ]

    # Print the indices of features above the threshold for each eigenvector
    for eigenvector_index, feature_indices in enumerate(indices_of_features_above_threshold):
        print(f"Eigenvector {eigenvector_index+1}: Indices of features above threshold: {feature_indices}")

    overlapping = is_overlapping(eigenvectors, indices_of_features_above_threshold)
    if (overlapping):
        return indices_of_features_above_threshold


def write_global_fitness_to_csv(global_fitness_list, target_directory, file_name):
    """
    Write global fitness values to a CSV file.
    """
    file_path = os.path.join(target_directory, file_name)
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        rows = [[fitness] for fitness in global_fitness_list]
        writer.writerows(rows)

def run_fea_process(data_file_path, target_directory, result_file_name, num_factors, fea_runs, generations, pop_size):
    """
    Function to setup and run the Factored Evolutionary Algorithm process.
    """
    # Load and preprocess data
    X_std = load_and_prepare_pca_data(data_file_path)

    # Perform PCA and extract factors
    factors =get_pca_factors(X_std, num_factors)
    # Printing the factors in a formatted way
    print("PCA Factors:")
    print(factors)

    selected_factors = get_selected_features(factors)
    

    # Define the factor architecture
    FactorArchitecture(dim=10, factors=selected_factors)

    # Define the objective function
    function = Function(function_number=1, lbound=-23, ubound=32)

    # # Instantiate and run the FEA
    # fea = FEA(
    #     function=function,
    #     fea_runs=fea_runs,
    #     generations=generations,
    #     pop_size=pop_size,
    #     factor_architecture=factor_architecture,
    #     base_algorithm=PSO
    # )
    # fea.run()

    # #Write results to CSV
    # write_global_fitness_to_csv(fea.global_fitness_list, target_directory, result_file_name)

def main():
    # Define file paths and parameters
    data_file_path = "/Users/xuyingwangswift/Documents/MLIndependentStudy/src/DataAnalysis/ackley.csv"
    target_directory = '/Users/xuyingwangswift/Documents/MLIndependentStudy/src/results'
    result_file_name = 'f1_data.csv'
    num_factors = 10
    # these vairables also need to be turned. 
    fea_runs = 2
    generations = 500
    pop_size = 10

    # Call the FEA process function
    run_fea_process(data_file_path, target_directory, result_file_name,  num_factors, fea_runs, generations, pop_size)

if __name__ == "__main__":
    main()


# def main():

#     # Define file paths and parameters
#     data_file_path = "/path/to/ackley.csv"
#     target_directory = '/path/to/results'
#     result_file_name = 'f1_data.csv'
#     num_factors = 10
#     top_n_variables = 10
#     fea_runs = 100
#     generations = 500
#     pop_size = 10


#     # Define the objective function
#     # Replace 'function_number' with the specific function you want to optimize

#     function = Function(function_number=1, lbound=-23, ubound=32)

#     df = pd.read_csv("/Users/xuyingwangswift/Documents/MLIndependentStudy/src/data/dimension10/ackley.csv")

#     X = df.drop('target', axis=1)
#     y = df['target']
#     # Standardizing the features
#     X_std = StandardScaler().fit_transform(X)
   
#     pca_full = PCA()
#     pca_full.fit(X_std)

#     # Extracting the eigenvectors
#     eigenvectors = pca_full.components_

#     # Analyzing PCA results to define factors
#     num_factors = 10  # Define the number of factors
#     top_n_variables = 10  # Number of top variables to include in each factor
#     factors = [np.argsort(np.abs(component))[-top_n_variables:] for component in eigenvectors[:num_factors]]


#     # in the context of the Factored Evolutionary Algorithm (FEA) based on the information provided, 
#     # a "factor" refers to a specific subset of variables from the entire set of problem variables. 
#     # These factors are crucial in the FEA as they define how the problem's variables are grouped and optimized within subpopulations or swarms.

#     # Each factor is essentially a group of variables that are optimized together. 
#     # In FEA, the optimization problem is decomposed into smaller, more manageable sub-problems, each corresponding to a factor. 
#     # By focusing on a subset of variables, these subpopulations can efficiently search for optimal or near-optimal solutions in their respective subspaces of the entire problem domain.

#     #Define the factor architecture
#     #You will need to replace this with your specific factor architecture setup
#     factor_architecture = FactorArchitecture(dim=10, factors=factors)
#     # Example: factor_architecture.load_csv_architecture("path_to_factor_architecture.csv")
#     print(factor_architecture.factors)

#     # covered_variables = set()
#     # for factor in factors:
#     #     covered_variables.update(factor)

#     # are_all_variables_covered = len(covered_variables) == 10
#     # print(are_all_variables_covered)

#     # Define parameters for the FEA
#     fea_runs = 100
#     generations = 500
#     pop_size = 10

#     # Instantiate the FEA with the objective function, factor architecture, and PSO as the base algorithm
#     fea = FEA(
#         function=function,
#         fea_runs=fea_runs,
#         generations=generations,
#         pop_size=pop_size,
#         factor_architecture=factor_architecture,
#         base_algorithm=PSO
#     )

#     # Run the FEA
#     fea.run()

#     global_fitness_list = fea.global_fitness_list

#     target_directory = '/Users/xuyingwangswift/Documents/MLIndependentStudy/src/results'
#     file_name = 'f1_data.csv'
#     file_path = os.path.join(target_directory, file_name)

#     with open(file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # Wrap each float in a list
#         rows = [[fitness] for fitness in global_fitness_list]
#         writer.writerows(rows)

# # Ensure this script block runs only when this script is executed, and not when imported
# if __name__ == "__main__":
#     main()
