# -*- coding: utf-8 -*-
import unittest
import integration
import numpy as np

class TestExpSinhQuadrature(unittest.TestCase):
    def test_exp_density(self):
        scheme = integration.ExpSinhQuadrature(0.5, 1e-8, 100)
        
        def exp_dist(x): return np.exp(-x)
        self.assertAlmostEqual(scheme(exp_dist), 1.)

if __name__ == '__main__':
    unittest.main()
