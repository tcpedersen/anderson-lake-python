# -*- coding: utf-8 -*-

class NoConvergenceError(Exception):
    def __init__(self, message):
        super().__init__(message)
