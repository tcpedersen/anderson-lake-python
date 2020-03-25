# Implementation of the Anderson-Lake pricing scheme for the Heston Stochastic Volatility model in Python
Full Python implementation of the Heston pricing scheme developed by Leif Anderson and Mark Lake in their article Robust High-Precision Option Pricing by Fourier Transforms: Contour Deformations and Double-Exponential Quadrature.

The file working-example.py contains a working example of how to use the two Anderson-Lake functions.
## Overview of files in project
1. working-example.py contains a working example of how to use the two functions **anderson_lake** and **anderson_lake_expsinh**.
2. pricers.py contains the function **anderson_lake** and the simpler version **anderson_lake_expsinh**, which computes the call option price in the Heston model. 
The file also includes a closed-form Black-Scholes formula **bs_call_option** and a Monte Carlo implementation of the Heston model **heston_monte_carlo** capable of calculating prices for any type of simple option.
3. models.py contains the class **HestonModel** used in the functions anderson_lake and heston_monte_carlo. 
4. options.py contains the class **EuropeanCallOption** used in all pricer functions.
5. integration.py contains the class **ExpSinhQuadrature** used by **anderson_lake** and implicitly in **anderson_lake_expsinh**. The class computes integrals over the positive real line.
