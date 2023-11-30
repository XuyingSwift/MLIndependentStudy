import os
import pickle
import numpy as np
from utilities.exceptions import PickleException

class FactorArchitecture:
    def __init__(self, dim=0, factors=None):
        """
        Initialize the FactorArchitecture.

        :param dim: Integer representing the number of dimensions or variables in the problem.
        :param factors: List of factors (subsets of variables). Each factor is a list of variable indices.
        """
        self.dim = dim
        self.factors = factors if factors is not None else []
        self.arbiters = []
        self.optimizers = []
        self.neighbors = []
        self.method = ""
        self.function_evaluations = 0

        if self.factors:
            self.get_factor_topology_elements()

    def save_architecture(self, path_to_save=""):
        """
        Save the current state of the FactorArchitecture to a file.

        :param path_to_save: String representing the file path to save the architecture. If not provided, a default path is used.
        """
        default_path = "factor_architecture_files"
        default_filename = f"{self.method}_{self.dim}.pickle"

        if not path_to_save:
            path_to_save = os.path.join(default_path, self.method, default_filename)
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

        with open(path_to_save, "wb") as file:
            pickle.dump(self.__dict__, file)

    def load_architecture(self, path_to_load="", method="", dim=0):
        """
        Load an existing FactorArchitecture from a file.

        :param path_to_load: String representing the full file path to the pickle file. If not provided, method and dim are used to construct the path.
        :param method: Method name used to construct the file path.
        :param dim: Dimension used to construct the file path.
        """
        if not path_to_load:
            if not method or not dim:
                raise PickleException("Method name and dimension are required if no file path is provided.")
            path_to_load = os.path.join("factor_architecture_files", method, f"{method}_{dim}.pickle")

        if not os.path.exists(path_to_load) or os.path.isdir(path_to_load):
            raise PickleException(f"Invalid file path: {path_to_load}")

        with open(path_to_load, "rb") as file:
            pickle_object = pickle.load(file)
            self.__dict__.update(pickle_object)

    def get_factor_topology_elements(self):
        """
        Calculate topology elements based on the current factors, including arbiters, optimizers, and neighbors.
        """
        self.calculate_arbiters()
        self.calculate_optimizers()
        self.determine_neighbors()
