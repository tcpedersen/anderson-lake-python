# -*- coding: utf-8 -*-
import numpy as np
from torch.quasirandom import SobolEngine

from scipy.optimize import brentq, fminbound
from scipy.stats import norm

# Local modules
from integration import ExpSinhQuadrature
from models import HestonModel, BlackScholesModel
from options import EuropeanCallOption

# ==============================================================================
# === Black and Scholes
def bs_option_price(model: BlackScholesModel, 
                  option: EuropeanCallOption) -> float:
    d1 = ( np.log(model.forward / option.strike) + \
          (model.vol**2 / 2) * option.tau ) / (model.vol * np.sqrt(option.tau))
    d2 = d1 - model.vol * np.sqrt(option.tau)
    
    return np.exp(-model.rate * option.tau) * (model.forward * norm.cdf(d1) - \
                  option.strike * norm.cdf(d2))

# ==============================================================================
# === Anderson-Lake Heston closed form pricing function
def anderson_lake_expsinh(model: HestonModel, 
                            option: EuropeanCallOption) -> float:
    scheme = ExpSinhQuadrature(init_step_size=0.5, error_tol=1e-12, 
                                max_iter=10000)
    return anderson_lake(model, option, scheme)


def anderson_lake(model: HestonModel, option: EuropeanCallOption, 
                  scheme: ExpSinhQuadrature) -> float:
    omega = calc_omega(model, option)
    phi = calc_phi(model, option)
    alpha = calc_alpha(model, option)

    # Define integrand
    tphi = np.tan(phi)
    tphip = 1 + 1j * tphi

    def Q(z): return model.cf(z - 1j, option.tau) / (z * (z - 1j))

    def integrand(x):
        dexp = np.exp(-x * tphi * omega + 1j * x * omega)
        return (dexp * Q(-1j * alpha + x * tphip) * tphip).real

    I = np.exp(alpha * omega) * scheme(integrand)
    R = model.forward * (alpha <= 0) - option.strike * (alpha <= -1) - \
        0.5 * (model.forward * (alpha == 0) - option.strike * (alpha == -1))
    return np.exp(- model.rate * option.tau) * (R - model.forward / np.pi * I)

# ==============================================================================
# === Helper functions
def calc_omega(model: HestonModel, option: EuropeanCallOption) -> float:
    return np.log(model.forward / option.strike)

def calc_phi(model: HestonModel, option: EuropeanCallOption) -> float:
    omega = calc_omega(model, option)
    r = model.rho - model.sigma * omega / \
        ( model.vol + model.kappa * model.theta * option.tau )
    if r * omega < 0:
        return np.pi / 12 * np.sign(omega)
    else:
        return 0

# ==============================================================================
# === Function for finding optimal alpha
def calc_alpha(model: HestonModel, option: EuropeanCallOption) -> float:
    omega = calc_omega(model, option)

    # The interval in which to locate alpha is technically an open interval,
    # so a small number is added/substracted to/from the boundaries.
    eps = np.sqrt(np.finfo(np.float64).eps)

    alpha_min, alpha_max = alpha_min_max(model, option)
    if omega >= 0:
        alpha, val = locate_optimal_alpha(model, option, alpha_min, -1 - eps)
    elif omega < 0 and model.kappa - model.rho * model.sigma > 0 \
        and abs(alpha_max) > 1e-3:
        alpha, val = locate_optimal_alpha(model, option, eps, alpha_max)
    else:
        alpha, val = locate_optimal_alpha(model, option, eps, alpha_max)
        if val > 9:
            alpha, val = locate_optimal_alpha(model, option,
                                              alpha_min, -1 - eps)
    return alpha


def locate_optimal_alpha(model, option, a, b):
    omega = np.log(model.forward / option.strike)
    obj_func = lambda alpha: model.log_cf_real(alpha, option.tau) -\
        np.log(alpha * (alpha + 1)) + alpha * omega

    alpha, val = fminbound(obj_func, a, b, full_output=True)[0:2]
    return alpha.real, val.real


def k_plus_minus(x: float, sign: int, model: HestonModel, 
                 option: EuropeanCallOption) -> float:
    A = model.sigma - 2 * model.rho * model.kappa
    B = (model.sigma - 2 * model.rho * model.kappa)**2 +\
        4 * (model.kappa**2 + x**2 / option.tau**2) * (1 - model.rho**2)
    C = 2 * model.sigma * (1 - model.rho**2)

    return (A + sign * np.sqrt(B)) / C


def critical_moments_func(k: float, model: HestonModel,
                          option: EuropeanCallOption) -> float:
    kminus = k_plus_minus(0, -1, model, option)
    kplus = k_plus_minus(0, 1, model, option)
    
    beta = model.kappa - model.rho * model.sigma * k
    D = np.sqrt(beta**2 + model.sigma**2 * (-1j * k) * ((-1j * k) + 1j))

    if k > kplus or k < kminus:
        D = abs(D)
        return np.cos(D * option.tau / 2) + \
            beta * np.sin(D * option.tau / 2) / D
    else:
        D = D.real
        return np.cosh(D * option.tau / 2) + \
            beta * np.sinh(D * option.tau / 2) / D


def alpha_min_max(model: HestonModel,
                  option: EuropeanCallOption) -> (float, float):
    kminus = k_plus_minus(0, -1, model, option)
    kplus = k_plus_minus(0, 1, model, option)
    
    # The interval in which to locate k is technically an open interval,
    # so a small number is added/substracted to/from the boundaries.
    eps = np.sqrt(np.finfo(np.float64).eps)
    
    # Find kmin
    kmin2pi = k_plus_minus(2 * np.pi, -1, model, option)
    kmin = brentq(critical_moments_func, kmin2pi + eps, kminus - eps,
                  args=(model, option))

    # Find kmax
    kps = model.kappa - model.rho * model.sigma
    if kps > 0:
        a, b = kplus, k_plus_minus(2 * np.pi, 1, model, option)
    elif kps < 0:
        T = -2 / (model.kappa - model.rho * model.sigma * kplus)
        if option.tau < T:
            a, b = kplus, k_plus_minus(np.pi, 1, model, option)
        else:
            a, b = 1, kplus
    else:
        a, b = kplus, k_plus_minus(np.pi, 1, model, option)
    kmax = brentq(critical_moments_func, a + eps, b - eps,
                  args=(model, option))

    return kmin - 1, kmax - 1

# ==============================================================================
# === Heston Monte Carlo
def heston_monte_carlo(model: HestonModel, option, num_steps, num_paths):
    # Generate brownian motion increments from a sobol sequence
    num_stochastic_processes = 2
    sobol = SobolEngine(num_stochastic_processes * num_steps)
    samples = norm.ppf(sobol.draw(num_paths))
    increment1, increment2 = np.hsplit(samples, num_stochastic_processes)

    kappa, theta, sigma, rho = model.kappa, model.theta, model.sigma, model.rho

    dt = 1 / num_steps

    ones = np.ones(num_paths).reshape(-1, 1)
    trunc_vol = ones * model.vol
    true_vol = trunc_vol
    moved_forward = ones * model.forward

    dw1 = np.array(np.sqrt(dt) * increment1)
    dw2 = np.array(rho * np.sqrt(dt) * increment1 + \
                   np.sqrt(1.0 - rho**2) * np.sqrt(dt) * increment2)

    for num_step in range(int(num_steps * option.tau)):
        moved_forward *= np.exp(np.sqrt(true_vol) * dw1[:, num_step, None])
        
        positive_part_trunc_vol = np.maximum(trunc_vol, 0)
        trunc_vol += kappa * (theta - positive_part_trunc_vol) * dt + \
            sigma * np.sqrt(positive_part_trunc_vol) * dw2[:, num_step, None]

        true_vol = np.maximum(trunc_vol, 0)

    return np.exp(- model.rate * option.tau) * np.mean(option(moved_forward))
