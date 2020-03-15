'# -*- coding: utf-8 -*-'
import numpy as np

class EuropeanCallOption:
    def __init__(self, tau, strike):
        self.tau = tau
        self.strike = strike
        
    def __call__(self, forward):
        return np.maximum(forward - self.strike, 0)

    def __str__(self):
        out_str = f"tau: {self.tau}\n\r" +\
                  f"strike: {self.strike}\n\r"
        return out_str
