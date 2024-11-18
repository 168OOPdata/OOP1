"""
Testing implementation for Task 1 requirements:
a) Implementation of three GLMs from Table 1
b) Using OOP concepts (inheritance, abstract methods, polymorphism, overriding)
c) Comparing parameter estimates β and predicted values μᵢ with statsmodels
d) Flexible testing interface using argparse
"""
import argparse # For parsing command-line arguments.
import numpy as np
import pandas as pd
import statsmodels.api as sm # For GLM implementation in statsmodels.
import matplotlib.pyplot as plt
from GLM_classes import NormalGLM, BernoulliGLM, PoissonGLM


def load_dataset(dset, predictors):
    """
    Load dataset and selects the predictors and response variable based on user input and return predictors (X) and response (y).

    Load appropriate dataset for each GLM type:
    -------
    Distributions and Link Functions (Table 1):
    - Normal GLM (duncan): 
        * Identity link: I(μ) = μ
        * Continuous response (income)
    
    - Bernoulli GLM (spector): 
        * Logit link: log(μ/(1-μ))
        * Binary response (GRADE)
    
    - Poisson GLM (warpbreaks): 
        * Log link: log(μ)
        * Count response (breaks)
    
    Parameters:
    dset (str): Dataset name - options are 'duncan', 'spector', 'warpbreaks'.
    predictors (list): List of predictor variable names.

    Returns:
    X (DataFrame): Predictor variables selected based on user input.
    y (Series): Response variable for the selected GLM type.
    """

    if dset == 'duncan':
        # Load dataset for Normal GLM
        data = sm.datasets.get_rdataset("Duncan", "carData").data
        y = data['income']  # Response variable 'y' for Normal GLM - expect the actual incomes to be continuous
    elif dset == 'spector':
        # Load dataset for Bernoulli GLM
        data = sm.datasets.spector.load_pandas().data
        y = data['GRADE']  # Response variable 'y' for Bernoulli GLM - expect the actual pass '1'/fail '0' results
    elif dset == 'warpbreaks':
        # Load dataset for Poisson GLM
        data = pd.read_csv('warpbreaks.csv')
        y = data['breaks']  # Response variable 'y' for Poisson GLM - expect the actual break counts to be non-negative integers
    else:
        raise ValueError("Unknown dataset.")

    # Validate that the specified predictors exist in the dataset
    for predictor in predictors:
        if predictor not in data.columns:
            raise ValueError(f"Predictor '{predictor}' not found in dataset.")

    # Select the predictors specified by the user!
    X = data[predictors]

    return X, y


def fit_and_compare(model, X, y): # 'model' parameter can be any subclass of GLM
    """
    Compare GLM implementations with statsmodels.
    
    Task 1(c) Requirements:
    1. Compare parameter estimates (β):
       - From our implemented GLM classes (NormalGLM, BernoulliGLM, PoissonGLM)
       - With statsmodels built-in GLM implementation
    
    2. Compare predicted values (μᵢ):
       - Using our implemented link functions from Table 1
       - With statsmodels predictions
    
    Parameters:
    -----------
    model : Class
        One of our implemented GLM classes (NormalGLM, BernoulliGLM, PoissonGLM)
    X : array-like
        Feature matrix with shape (n_samples, n_features)
    y : array-like
        Target values
    """
    # Fit our implemented GLM
    implemented_model = model(X, y)
    implemented_model.fit()
    implemented_params = implemented_model.params # Implemented β
    implemented_preds = implemented_model.predict() # Implemented µ

    # Fit statsmodels GLM for comparison
    family_dict = {
        NormalGLM: sm.families.Gaussian(),   # For Normal GLM
        BernoulliGLM: sm.families.Binomial(),  # For Bernoulli GLM
        PoissonGLM: sm.families.Poisson()    # For Poisson GLM
    }

    # Instantiate and fit the statsmodels GLM with the same predictors and response
    sm_model = sm.GLM(y, X, family=family_dict[type(implemented_model)])
    sm_results = sm_model.fit() # Getting 'Statsmodels β' 
    sm_preds = sm_results.predict(X) # Getting 'Statsmodels μ' (Reference Predictions)

    # Output results for comparison
    print("\nParameter Estimates (β):")
    print("Implemented GLM:", implemented_params)
    print("Statsmodels GLM:", sm_results.params.values)

    print("\nPredicted Values (μᵢ) - First 5 observations:")
    print("Implemented GLM:", implemented_preds[:5])
    print("Statsmodels GLM:", sm_preds[:5])

    # Plot results for visual comparison
    plt.figure(figsize=(10, 6))
    plt.plot(y.values, label="Observed Values", marker='o', linestyle='-', color='black')
    plt.plot(implemented_preds, label="Implemented GLM", marker='x', linestyle='--', color='blue')
    plt.plot(sm_preds, label="Statsmodels GLM", marker='s', linestyle=':', color='red')
    plt.xlabel("Observation Index")
    plt.ylabel("Response Value")
    plt.title(f"Comparison of {type(implemented_model).__name__} Predictions")
    plt.legend()
    plt.show()


def main():
    """
    Task 1(d): Command line interface using argparse.
    
    This function provides:
    1. Model Selection:
       - Normal GLM (Identity link)
       - Bernoulli GLM (Logit link)
       - Poisson GLM (Log link)

    2. Dataset Selection:
       - duncan: For Normal GLM
       - spector: For Bernoulli GLM
       - warpbreaks: For Poisson GLM

    3. Feature Selection:
       - Specify predictor variables via command-line arguments
       - Option to add intercept term

    Example Usage:
    --------------
    # For Normal GLM:
    python testing_glm.py --model normal --dset duncan --predictors education prestige --add_intercept

    # For Bernoulli GLM:
    python testing_glm.py --model bernoulli --dset spector --predictors GPA TUCE PSI --add_intercept

    # For Poisson GLM:
    python testing_glm.py --model poisson --dset warpbreaks --predictors wool tension --add_intercept
    """

    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Test GLM implementations from Table 1",
        epilog="Example: python testing_glm.py --model normal --dset duncan --predictors education prestige --add_intercept"
    )

    # Add arguments for model type, dataset, predictors, and intercept
    parser.add_argument("--model",
                        choices=['normal', 'bernoulli', 'poisson'],
                        required=True,
                        help="GLM type from Table 1")
    parser.add_argument("--dset",
                        choices=['duncan', 'spector', 'warpbreaks'],
                        required=True,
                        help="Dataset matching GLM type")
    parser.add_argument("--predictors",
                        nargs='+',
                        required=True,
                        help="Predictor variables")
    parser.add_argument("--add_intercept",
                        action="store_true",
                        help="Add intercept term to X")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load and prepare data
    X, y = load_dataset(args.dset, args.predictors)  # Load data using specified predictors

    # Add intercept term if specified by the user
    if args.add_intercept:
        X = sm.add_constant(X) # This transforms X to include intercept

    # Select appropriate GLM based on user input
    model_dict = {
        'normal': NormalGLM,
        'bernoulli': BernoulliGLM,
        'poisson': PoissonGLM
    }
    model_class = model_dict[args.model]

    # Fit the model and compare with statsmodels
    fit_and_compare(model_class, X, y)

if __name__ == "__main__":
    main()






 