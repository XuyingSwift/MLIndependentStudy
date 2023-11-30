import numpy as np
import csv
import os
from benchmark_functions import ackley
from benchmark_functions import brown
from benchmark_functions import dixon_price
from benchmark_functions import exponential
from benchmark_functions import griewank
from benchmark_functions import powell_singular
from benchmark_functions import rana
from benchmark_functions import rastrigin
from benchmark_functions import schwefel
from benchmark_functions import sphere
from benchmark_functions import rosenbrock


def generate_function_data(dimensions, num_samples, dir_path):
    # Define a list of benchmark functions and their corresponding input domains
    benchmark_functions = [
        (ackley,           (-32, 32)),
        (brown,              (-1, 4)),
        (dixon_price,      (-10, 10)),
        (exponential,        (-1, 1)),
        (griewank,       (-100, 100)),
        (powell_singular,    (-4, 5)),
        (rana,           (-500, 500)),
        (rastrigin,    (-5.12, 5.12)),
        (rosenbrock, (-2.048, 2.048)),
        (schwefel,       (-512, 512)),
        (sphere,       (-5.12, 5.12))
    ]

    all_results = []

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for func, domain in benchmark_functions:
        results = []

        for _ in range(num_samples):
            input_sample = np.random.uniform(domain[0], domain[1], dimensions)
            target = func(input_sample)

            # Store input sample and result as a tuple
            results.append((input_sample.tolist(), target))

        # Append results for the current function to the overall results list
        all_results.append(results)

        # Specify the file path where to save the CSV file
        csv_file_path = os.path.join(dir_path, f"{func.__name__}.csv")

        # Write the results to the CSV file
        with open(csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row with variable names
            csv_writer.writerow([f"x{i}" for i in range(1, dimensions + 1)] + ["target"])

            # Write the data rows
            for input_sample, result in results:
                csv_writer.writerow(input_sample + [result])

        print(f"Results for {func.__name__} saved to {csv_file_path}")
