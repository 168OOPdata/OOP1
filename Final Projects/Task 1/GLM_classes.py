"""
Generalized Linear Models (GLMs)

This implementation follows the theoretical foundation:
1. Linear predictor: ηᵢ = xᵢᵀβ
2. Link function: g(μᵢ) = ηᵢ
3. Maximum likelihood estimation

Table 1 Link Functions:
- Normal: Identity I(μ)
- Bernoulli: Logit log(μ/(1-μ))
- Poisson: Log log(μ)
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize # Find the parameters that minimize the negative log-likelihood.
from scipy.stats import norm, bernoulli, poisson

# Superclass GLM
class GLM(ABC):
    """
    Abstract base class for GLMs following the mathematical formulations:

    Following equation (1): ηᵢ = β₀ + β₁xᵢ,₁ + β₂xᵢ,₂ + ... + βₚxᵢ,ₚ = xᵢᵀβ
    where:
    - β = (β₀, β₁, ..., βₚ)ᵀ are unknown parameters
    - xᵢ = (1, xᵢ,₁, xᵢ,₂, ..., xᵢ,ₚ)ᵀ are covariates
    - ηᵢ is the linear predictor
    
    Link function (equation 2): g(μᵢ) = ηᵢ
    """
    def __init__(self, X, y):
        # Ensure X and y have compatible dimensions
        assert len(X) == len(y), "Predictors (X) and response (y) must have the same number of observations."
        self.X = X # X includes intercept (1, xᵢ,₁, xᵢ,₂, ..., xᵢ,ₚ)
        self.y = y
        self.params = None  # β parameters to be estimated

    @abstractmethod # Required to be implemented by all subclasses
    def fit(self):
        """Abstract method to fit model parameters via maximum likelihood estimation."""
        pass

    @abstractmethod
    def predict(self, X=None):
        """Abstract method to make predictions based on fitted parameters."""
        pass

# Subclass for Normal Distribution GLM
class NormalGLM(GLM):
    """
    GLM subclass: Normal GLM with identity link: I(μ) = μ
    The identity function I(μ) means it returns exactly what you input, so:I(μ) = μ, Therefore, g(μ) = I(μ) = μ
    Normal GLM:
    - Linear predictor: ηᵢ = xᵢᵀβ
    - Link function: I(μ) = μ
    - Therefore: μᵢ = ηᵢ = xᵢᵀβ
    """

    def __init__(self, X, y):
        super().__init__(X, y)

    def fit(self):
        # Initialize parameters with small values
        initial_params = np.full(self.X.shape[1], 0.01)

        def neg_log_likelihood(params):
            eta = np.dot(self.X, params) # ηᵢ = xᵢᵀβ
            mu = eta  # Identity link: μᵢ = ηᵢ
            return -np.sum(norm.logpdf(self.y, mu))

        # Optimize parameters β using MLE
        result = minimize(neg_log_likelihood, initial_params)
        self.params = result.x # This is 'Implemented β'

    def predict(self, X=None):
        if X is None:
            X = self.X
        eta = np.dot(X, self.params) # Calculate η = xᵢᵀβ
        return eta  # Identity link function, μ = η (inverse link)


# Subclass for Bernoulli Distribution GLM
class BernoulliGLM(GLM):
    """GLM subclass: Bernoulli GLM with logit link: log(μ/(1-μ))
    Bernoulli GLM:
    - Linear predictor: ηᵢ = xᵢᵀβ
    - Link function: log(μ/(1-μ))
    - Therefore: μᵢ = 1/(1 + e^(-xᵢᵀβ))
    """

    def __init__(self, X, y):
        super().__init__(X, y)

    def fit(self):
        # Initialize parameters with small values
        initial_params = np.full(self.X.shape[1], 0.01)

        def neg_log_likelihood(params):
            eta = np.dot(self.X, params)
            mu = 1 / (1 + np.exp(-eta))  # Logit link
            return -np.sum(bernoulli.logpmf(self.y, mu))

        # Optimize parameters using MLE
        result = minimize(neg_log_likelihood, initial_params) 
        self.params = result.x

    def predict(self, X=None):
        if X is None:
            X = self.X
        eta = np.dot(X, self.params) # Calculate η
        return 1 / (1 + np.exp(-eta))  # Logit link function, μ = 1/(1 + e⁻η) (inverse link)


# Subclass for Poisson Distribution GLM
class PoissonGLM(GLM):
    """GLM subclass: Poisson GLM with log link: log(μ)
    Poisson GLM:
    - Linear predictor: ηᵢ = xᵢᵀβ
    - Link function: log(μ)
    - Therefore: μᵢ = exp(xᵢᵀβ)
    """

    def __init__(self, X, y):
        super().__init__(X, y)

    def fit(self):
        # Initialize parameters with small values
        initial_params = np.full(self.X.shape[1], 0.01)

        def neg_log_likelihood(params):
            eta = np.dot(self.X, params)
            mu = np.exp(eta)  # Log link
            return -np.sum(poisson.logpmf(self.y, mu))

        # Optimize parameters using MLE
        result = minimize(neg_log_likelihood, initial_params)
        self.params = result.x

    def predict(self, X=None):
        if X is None:
            X = self.X
        eta = np.dot(X, self.params) # Calculate η
        return np.exp(eta)  # Log link function, μ = exp(η) (inverse link)