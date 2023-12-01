import pandas as pd
import matplotlib.pyplot as plt

benchmark_functions = [
    'ackley',
    'dixon_price',
    'exponential',
    'griewank',
    'powell_singular',
    'rana',
    'rastrigin',
    'rosenbrock',
    'schwefel',
    'sphere'
]

# Base directory for your files
base_dir_dim10 = "/Users/xswift/Desktop/MLIndependentStudy/src/PCA_FEA/results/dim10_gen10000/"
file_end_dim10 = "_data_dim_10_gen_10000_result.csv"

base_dir_dim30 = "/Users/xswift/Desktop/MLIndependentStudy/src/PCA_FEA/results/dim30_gen25000/"
file_end_dim30 = "_data_dim_30_gen_25000_result.csv"

for function_name in benchmark_functions:
    # Construct the CSV file path
    csv_file = f"{base_dir_dim10}performance_result/{function_name}{file_end_dim10}"
    title = f"FEA Performance Over Runs on {function_name} with Dim10"
    plot_name = f"{base_dir_dim10}performance_plot/{function_name}_dim10.png"

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Assuming the CSV file has columns named 'FEA_Run' and 'Global_Fitness'
    # If they have different names, replace them with the actual column names
    fea_runs = df['FEA Run']
    global_fitness = df['Global Fitness']

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(fea_runs, global_fitness, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('FEA Run')
    plt.ylabel('Global Fitness')
    plt.grid(True)
    # Save the plot as a PNG file
    plt.savefig(plot_name)
    # Clear the current figure after saving it
    plt.clf()
