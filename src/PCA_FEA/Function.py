import numpy as np
import math

class Function():
    def __init__(self, function_number, lbound, ubound):
        self.dimensions = 0
        self.lbound = lbound
        self.ubound = ubound
        self.function_to_call = 'F'+str(function_number)

        function_names = {
            1: "ackley",
            2: "dixon_price",
            3: "exponential",
            4: "griewank",
            5: "powell_singular",
            6: "rana",
            7: "rastrigin",
            8: "rosenbrock",
            9: "schwefel",
            10: "sphere"
        }

        self.name = function_names.get(function_number, "")
    
    def __str__(self):
        """
        Provides a string representation of the Function object.

        :return: A string detailing the attributes of the Function.
        """
        # ANSI escape codes for colors
        header_color = '\033[95m'  # Purple
        key_color = '\033[94m'  # Blue
        value_color = '\033[92m'  # Green
        end_color = '\033[0m'  # Reset color

        description = f"{header_color}Function Details:{end_color}\n"
        description += f"  {key_color}Name:{end_color} {value_color}{self.name}{end_color}\n"
        description += f"  {key_color}Lower Bound:{end_color} {value_color}{self.lbound}{end_color}\n"
        description += f"  {key_color}Upper Bound:{end_color} {value_color}{self.ubound}{end_color}\n"
        description += f"  {key_color}Function to Call:{end_color} {value_color}{self.function_to_call}{end_color}\n"
        return description

    def run(self, solution):
        """
        Executes the optimization function specified in 'self.function_to_call' on the provided solution.

        This method is the central point for computing the fitness of a solution based on the selected optimization function. 
        It dynamically calls the appropriate function within this class based on the 'self.function_to_call' attribute.

        :param solution: A numpy array or list representing a candidate solution to the optimization problem.
        :return: The computed fitness value of the provided solution.
        """

        # If the problem dimensions haven't been set yet (i.e., dimensions == 0),
        # determine and set the dimensions based on the length of the provided solution.
        if self.dimensions == 0:
            self.dimensions = len(solution)
            # Optionally, here you can include a check for the problem size to ensure it's within expected limits.

        # Dynamically call the function specified in 'self.function_to_call'.
        # 'getattr' is used to retrieve the function by its name (as a string) and then call it with the solution.
        # This approach provides flexibility, allowing the class to easily switch between different optimization functions.
        return getattr(self, self.function_to_call)(solution=solution)



        # function 1
    def F1(self, solution, name = "ackley"):
        self.name = name
        # Define the constants a, b, and c for the Ackley function
        a = 20
        b = 0.2
        c = 2 * math.pi
        # Get the number of dimensions (n) from the length of the input array x
        n = len(solution)
        # Calculate the first part of the Ackley function
        sum1 = np.sum(solution ** 2)
        # Calculate the second part of the Ackley function
        sum2 = np.sum(np.cos(c * solution))
        # Calculate the individual terms for the Ackley function
        term1 = -a * math.exp(-b * math.sqrt(sum1 / n))
        term2 = -math.exp(sum2 / n)
        # Combine the terms and add constants to get the final result
        result = term1 + term2 + a + math.exp(1)
        
        return result

    # # function 2
    # def F2(self, solution, name="brown"):
    #     self.name = name
    #     # Extract the elements of the input vector x except the last one (xi)
    #     xi = solution[:-1]
    #     # Extract the elements of the input vector x starting from the second element (xi+1)
    #     xi1 = solution[1:]
    #     # Calculate the terms for each pair of xi and xi+1
    #     with np.errstate(over='ignore'):
    #         terms = ((xi ** 2) ** ((xi ** 2 + xi1 + 1) / (xi ** 2 + xi1)))
    #     # Sum up the terms and add the first element of the input vector (x0)
    #     result = terms.sum() + solution[0]
    #     # Return the final result
    #     return result

    # function 3
    def F2(self, solution, name="dixon_price"):
        self.name = name
        n = len(solution) 
        # Check the dimension of the input vector
        if n < 1:
            raise ValueError("Dimension of the input vector must be at least 1.")
        result = (solution[0] - 1) ** 2
        for i in range(1, n):
            result += (i + 1) * (2 * solution[i] ** 2 - solution[i - 1]) ** 2
        
        return result

    # function 4
    def F3(self, solution, name="exponential"):
        self.name = name
        return -np.exp(np.sum(np.square(solution)) * (-0.5))

    # function 5
    # The Griewank function has many widespread local minima, which are regularly distributed.
    def F4(self, solution, name="griewank"):
        self.name = name
        n = len(solution)
        sum_sq = np.sum(np.square(solution))
        prod_cos = np.prod(np.cos(solution / np.sqrt(np.arange(1, n + 1))))
        result = 1 + (sum_sq / 4000) - prod_cos
        return result

    # function 6
    def F5(self, solution, name="powell_singular"):
        self.name = name
        n = len(solution)
        if n < 4:
            raise ValueError("Dimension of the input vector must be at least 4.")
        result = 0
        for i in range(0, n - 3, 4):
            result += (solution[i] + 10 * solution[i + 1]) ** 2
            result += 5 * (solution[i + 2] - solution[i + 3]) ** 2
            result += (solution[i + 1] - 2 * solution[i + 2]) ** 4
            result += 10 * (solution[i] - solution[i + 3]) ** 4
        return result

    # function 7
    def F6(self, solution, name="rana"):
        self.name = name
        x1 = solution[0]
        xi = solution[1:]

        sqrt_abs_diff = np.sqrt(np.abs(x1 - xi + 1))
        sqrt_abs_sum = np.sqrt(np.abs(x1 + xi + 1))

        term1 = xi * np.sin(sqrt_abs_diff) * np.cos(sqrt_abs_sum)
        term2 = (x1 + 1) * np.sin(sqrt_abs_sum) * np.cos(sqrt_abs_diff)

        return np.sum(term1 + term2)

    # function 8
    def F7(self, solution, name="rastrigin"):
        self.name = name
        A = 10
        n = len(solution)
        if n < 1:
            raise ValueError("Dimension of the input vector must be at least 1.")
        sum_term = np.sum(solution**2 - A * np.cos(2 * np.pi * solution))
        result = A * n + sum_term
        
        return result

    # function 9
    def F8(self, solution, name="rosenbrock"):
        self.name = name
        n = len(solution)
        
        if n < 1:
            raise ValueError("Dimension of the input vector must be at least 1.")
        
        result = 0
        
        for i in range(n - 1):
            result += 100 * (solution[i + 1] - solution[i]**2)**2 + (1 - solution[i])**2
        
        return result

    # function 10 
    def F9(self, solution, name="schwefel"):
        self.name = name
        n = len(solution)
        
        if n < 1:
            raise ValueError("Dimension of the input vector must be at least 1.")
        
        result = 418.9829 * n
        
        for i in range(n):
            result -= solution[i] * np.sin(np.sqrt(abs(solution[i])))
        
        return result

    # function 11
    def F10(self, solution, name="sphere"):
        self.name = name
        n = len(solution)
        
        if n < 1:
            raise ValueError("Dimension of the input vector must be at least 1.")
        
        result = np.sum(solution**2)
        
        return result
    
            