# -*- coding: utf-8 -*-
import unittest
import numpy as np

from models import BlackScholesModel
from options import EuropeanCallOption
from pricers import bs_option_price

class TestBlackScholes(unittest.TestCase):
    def test_itm(self):
        model = BlackScholesModel(300 * np.exp(0.03 * 1), 0.15, 0.03)
        option = EuropeanCallOption(1, 250)
        
        result = bs_option_price(model, option)
        expected = 58.82
    
        self.assertAlmostEqual(result, expected, 2)

    
if __name__ == '__main__':
    unittest.main()
