import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import csv

# Example list of global best fitness values from each run of FEA
file_name = "/Users/xuyingwangswift/Documents/MLIndependentStudy/src/results/f1_data.csv"
global_best_fitnesses = []

# Open the CSV file for reading
with open(file_name, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV
    for row in csv_reader:
        # Append each row to the list
        global_best_fitnesses.append(row)

# 1. Calculate Basic Statistics
mean_fitness = np.mean(global_best_fitnesses)
median_fitness = np.median(global_best_fitnesses)
std_dev_fitness = np.std(global_best_fitnesses)
min_fitness = np.min(global_best_fitnesses)
max_fitness = np.max(global_best_fitnesses)

print("Mean Fitness:", mean_fitness)
print("Median Fitness:", median_fitness)
print("Standard Deviation:", std_dev_fitness)
print("Min Fitness:", min_fitness)
print("Max Fitness:", max_fitness)

# 2. Visualization (Histogram)
plt.hist(global_best_fitnesses, bins=10, alpha=0.7, color='blue')
plt.title('Distribution of Global Best Fitness')
plt.xlabel('Fitness')
plt.ylabel('Frequency')
plt.show()

# If you have more detailed data per iteration, you can plot convergence over iterations for a few runs.

# 3. Consistency and Variability Analysis
# Coefficient of Variation (CV) = (Standard Deviation / Mean) * 100
cv = (std_dev_fitness / mean_fitness) * 100
print("Coefficient of Variation:", cv, "%")

# 4. Statistical Significance
# This part is relevant only if you're comparing different configurations or algorithms.
# Example: t-test against another set of results
# other_algorithm_fitnesses = [/* fitness values from another algorithm */]
# t_stat, p_value = stats.ttest_ind(global_best_fitnesses, other_algorithm_fitnesses)
# print("T-test P-value:", p_value)

# Interpretation of results depends on your specific problem and objectives.
