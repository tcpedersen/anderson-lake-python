# -*- coding: utf-8 -*-
class EuropeanCallOption:
    def __init__(self, tau, strike):
        self.tau = tau
        self.strike = strike

    def __str__(self):
        out_str = f"tau: {self.tau}\n\r" +\
                  f"strike: {self.strike}\n\r"
        return out_str
