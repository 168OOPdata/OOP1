"""
Purpose:
1. The goal is to create a modular and reusable system for loading datasets using OOP principles. 
2. The superclass defines common behavior, while subclasses specialize 
for specific file formats (e.g., statsmodels, local CSVs, and online CSVs).

Key OOP Concepts Used:
1. Inheritance: Common functionality resides in the `DataLoader` superclass.
2. Overriding: Subclasses provide specialized behavior for `load_data`.
3. Polymorphism: Uniform interface across subclasses for `load_data`, `add_constant`, etc.
4. Decorators: Used for accessor methods to ensure safe and controlled access.
5. Assertions: Used to validate data integrity.
"""

import pandas as pd
import statsmodels.api as sm
from abc import ABC, abstractmethod


# Superclass
class DataLoader(ABC):
    """
    Purpose:
    - Define a common interface for all data loaders
    - Provide shared functionality (e.g., adding a constant, validating data)
    
    Key Methods:
    - load_data: Abstract method to be implemented by subclasses
    - add_constant: Adds an intercept term to the predictor matrix `X`
    - x: Property method to return predictors
    - y: Property method to return the response
    - x_transpose: Returns the transpose of the predictor matrix `X`
    """
    def __init__(self):
        self._data = None  # Placeholder for the loaded dataset

    @abstractmethod
    def load_data(self, **kwargs):
        """
        Abstract method to load data
        To be implemented by subclasses
        """
        pass

    def add_constant(self):
        """
        Add a constant (intercept) to the predictor matrix `X`
        Ensures the matrix matches the required statsmodels' dimensions [N, p+1]
       
        Equation 1: ηᵢ = β₀ + β₁xᵢ,₁ + β₂xᵢ,₂ + ... + βₚxᵢ,ₚ = xᵢᵀβ
        Original X has shape [p+1, N] (p+1 rows, N columns), and each xᵢ is a column in X.

        Statsmodels requirement:
        X needs shape [N, p+1] (N rows, p+1 columns), and each xᵢ is a row in X.

        """
        assert self._data is not None, "No data loaded. Call `load_data` first."
        self._data['constant'] = 1 # Include the intercept column equals to '1'

    # @property decorator ensures controlled access to predictors (x) and response (y). 
    @property 
    def x(self):
        """
        Accessor for predictor matrix `X`
        Includes all columns except the response
        """
        assert self._data is not None, "No data loaded. Call `load_data` first."
        return self._data.drop(columns=['response']).values

    @property 
    def y(self):
        """
        Accessor for response vector `y`
        Assumes a column named 'response' exists
        """
        assert self._data is not None, "No data loaded. Call `load_data` first."
        return self._data['response'].values

    def x_transpose(self):
        """
        Returns the transpose of the predictor matrix `X`
        """
        return self.x.T


# Subclass: Statsmodels Data Loader
class StatsmodelsLoader(DataLoader):
    """ 
    Purpose:
    - Load predefined datasets from statsmodels.
    - Example datasets: `Duncan`, `Spector`.
    
    Overrides:
    - load_data: Fetches data from statsmodels.
    """
    def load_data(self, dataset_name, package_name=None):
        """
        Parameters:
        - dataset_name (str): Name of the dataset (e.g., 'Duncan').
        - package_name (str, optional): Name of the package (e.g., 'carData').
        """
        try:
            if package_name:
                self._data = sm.datasets.get_rdataset(dataset_name, package_name).data
            else:
                self._data = sm.datasets.__getattribute__(dataset_name).load_pandas().data
            self._data.rename(columns={self._data.columns[-1]: 'response'}, inplace=True)
        except Exception as e:
            raise ValueError(f"Error loading statsmodels dataset: {e}")


# Subclass: Local CSV Data Loader
class CSVLoader(DataLoader):
    """
    Purpose:
    - Load datasets stored as CSV files on the local disk.
    
    Overrides:
    - load_data: Reads data from a local file path.
    """
    def load_data(self, file_path):
        """
        Parameters:
        - file_path (str): Path to the CSV file.
        """
        try:
            self._data = pd.read_csv(file_path)
            self._data.rename(columns={self._data.columns[-1]: 'response'}, inplace=True)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")


# Subclass: Internet CSV Data Loader
class InternetCSVLoader(DataLoader):
    """
    Purpose:
    - Load datasets stored as CSV files from a URL.
    
    Overrides:
    - load_data: Reads data from a URL.
    """
    def load_data(self, url):
        """
        Parameters:
        - url (str): URL of the CSV file.
        """
        try:
            self._data = pd.read_csv(url)
            self._data.rename(columns={self._data.columns[-1]: 'response'}, inplace=True)
        except Exception as e:
            raise ValueError(f"Error loading CSV from URL: {e}")


# Testing the Implementation
if __name__ == "__main__":
    # Test Statsmodels Loader
    print("\nStatsmodels Data Loader:")
    stats_loader = StatsmodelsLoader()
    stats_loader.load_data(dataset_name="Duncan", package_name="carData")
    stats_loader.add_constant()
    print(f"Original X shape: {stats_loader.x.shape}")
    print(f"Transposed X shape: {stats_loader.x_transpose().shape}")
    print("\nStatsmodels data:")
    print("X matrix (first 3 rows):")
    print(stats_loader.x[:3])
    print("y values (first 3 values):")
    print(stats_loader.y[:3])

    # Test CSV Loader
    print("\nCSV Data Loader:")
    csv_loader = CSVLoader()
    csv_loader.load_data("warpbreaks.csv")
    csv_loader.add_constant()
    print(f"Original X shape: {csv_loader.x.shape}")
    print(f"Transposed X shape: {csv_loader.x_transpose().shape}")
    print("\nStatsmodels data:")
    print("X matrix with constant (first 3 rows):")
    print(csv_loader.x[:3])
    print("y values (first 3 values):")
    print(csv_loader.y[:3])

    # Test Web Loader
    print("\nWeb Data Loader:")
    url = "https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv"
    web_loader = InternetCSVLoader()
    web_loader.load_data(url)
    web_loader.add_constant()
    print(f"Original X shape: {web_loader.x.shape}")
    print(f"Transposed X shape: {web_loader.x_transpose().shape}")
    print("\nStatsmodels data:")
    print("X matrix with constant (first 3 rows):")
    print(web_loader.x[:3])
    print("y values (first 3 values):")
    print(web_loader.y[:3])
