# -*- coding: utf-8 -*-
import unittest
import numpy as np

from pricers import anderson_lake
from integration import ExpSinhQuadrature
from models import HestonModel
from options import EuropeanCallOption

class TestAndersonLake(unittest.TestCase):
    # All expected prices found at https://kluge.in-chemnitz.de/tools/pricer/
    
    def setUp(self):
        self.scheme = ExpSinhQuadrature(0.5, 1e-12, 10000)
    
    def test_atm(self):
        model = HestonModel(100, 0.1197**2, 1.98937, 0.108977**2, 0.33147, \
                            0.0258519, 0)
        option = EuropeanCallOption(1, 100)
        
        result = anderson_lake(model, option, self.scheme)
        expected = 4.170956582
    
        self.assertAlmostEqual(result, expected)
    
    def test_itm(self):
        model = HestonModel(121.17361017736597, 0.1197**2, 1.98937, \
                            0.108977**2, 0.33147, -0.5, np.log(1.0005))
        option = EuropeanCallOption(0.50137, 150)
        
        result = anderson_lake(model, option, self.scheme)
        expected = 0.008644233552
        self.assertAlmostEqual(result, expected)
    
    def test_otm(self):
        model = HestonModel(11, 0.2**2, 2.0, 0.2**2, 0.3, -0.8, 0)
        option = EuropeanCallOption(2., 10)
        result = anderson_lake(model, option, self.scheme)
        expected = 1.7475020828 # result from finite difference method
        
        self.assertAlmostEqual(result, expected, 3)


if __name__ == '__main__':
    unittest.main()
