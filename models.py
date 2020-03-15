# -*- coding: utf-8 -*-
import numpy as np

NUMPY_COMPLEX128_MAX = np.finfo(np.complex128).max
NUMPY_LOG_COMPLEX128_MAX = np.log(NUMPY_COMPLEX128_MAX)

class HestonModel:
    def __init__(self, forward, vol, kappa, theta, sigma, rho, rate):
        self.forward = forward
        self.vol = vol
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.rate = rate

    def __str__(self):
        out_str = f"forward: {self.forward}\n\r" +\
                  f"vol: {self.vol}\n\r" +\
                  f"kappa: {self.kappa}\n\r" +\
                  f"theta: {self.theta}\n\r" +\
                  f"sigma: {self.sigma}\n\r" +\
                  f"rho: {self.rho}\n\r" + \
                  f"rate: {self.rate}\n\r"
        return out_str

    def cf(self, z, tau) -> complex:
        beta = self.kappa - 1j * self.sigma * self.rho * z
        D = np.sqrt(beta**2 + self.sigma**2 * z * (z + 1j))

        if beta.real * D.real + beta.imag * D.imag > 0:
            r = - self.sigma**2 * z * (z + 1j) / (beta + D)
        else:
            r = beta - D

        if D != 0:
            y = np.expm1(-D * tau) / (2 * D)
        else:
            y = -tau / 2

        A = self.kappa * self.theta / self.sigma**2 * \
            (r * tau - 2 * np.log1p(- r * y))
        
        B = z * (z + 1j) * y / (1 - r * y)
        
        exponent = A + B * self.vol
        
        if exponent > NUMPY_LOG_COMPLEX128_MAX:
            raise OverflowError("too large exponent in characteristic function")
        
        return np.exp(exponent)

    def log_cf_real(self, alpha, tau) -> float:
        # Evaluation of ln HestomModel.cf(-1j * (1 + alpha))
        beta = self.kappa - self.rho * self.sigma * (1 + alpha)
        Dsq = beta**2 - self.sigma**2 * (1 + alpha) * alpha
        
        if Dsq > 0:
            D = np.sqrt(Dsq)
            coshdt = np.cosh(D * tau / 2)
            sinhdt = np.sinh(D * tau / 2) / D
            nume = coshdt + beta * sinhdt
            
        else:
            # D = 1j * x
            x = np.sqrt(-Dsq)
            coshdt = np.cos(x * tau / 2)
            sinhdt = np.sin(x * tau / 2) / x
            nume = coshdt + beta * sinhdt

        A = self.kappa * self.theta / self.sigma**2 *\
            (beta * tau - np.log(nume**2))
        B = alpha * (1 + alpha) * sinhdt / nume

        return A + B * self.vol

class BlackScholesModel():
    def __init__(self, forward, vol, rate):
        self.forward = forward
        self.vol = vol
        self.rate = rate

    def cf(self, z, tau):
        return np.exp(-0.5 * self.vol * tau * z * (z + 1j))
