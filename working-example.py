# -*- coding: utf-8 -*-

# import local modules
from pricers import anderson_lake, anderson_lake_expsinh
from integration import ExpSinhQuadrature, GaussianQuadrature
from models import HestonModel
from options import EuropeanCallOption

# ==============================================================================
# === Example using the simpler function anderson_lake_expsinh

# Define the model
# The parameters are in order: 
# forward, initial volatility, kappa, theta, sigma, rho, interest rate
model = HestonModel(100, 0.1197**2, 1.98937, 0.108977**2, 0.33147, 0.0258519, 0)

# Define the call option
# The arguments are in order: 
# time to maturity, strike
option = EuropeanCallOption(1, 100)

# Calculate the price (should return 4.171)
price = anderson_lake_expsinh(model, option)
print(f"Price of option using anderson_lake_expsinh: {price}.")

# ==============================================================================
# === Example using the less simple function anderson_lake
# Define the quadrature method the function should use to integrate. Here the 
# one suggested in the Anderson-Lake article is used. It is implemented in 
# integration.py
scheme = GaussianQuadrature(1e-12, 1e-12, 1000)

# Define the model and option as in the simple case.
model = HestonModel(100, 0.1197**2, 1.98937, 0.108977**2, 0.33147, 0.0258519, 0)
option = EuropeanCallOption(1, 100)

# Calculate the price (should return 4.171)
price = anderson_lake(model, option, scheme)
print(f"Price of option using anderson_lake: {price}.")
