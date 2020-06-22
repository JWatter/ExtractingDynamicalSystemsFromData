import numpy as np

class LinearApprox:
    def __init__(self):
        self.coe = None
        self.bias = None
    
    def fit(self, x, y):
        self.coe = np.linalg.lstsq(x, y)[0]
        self.bias = np.linalg.lstsq(x, y)[1]

    def predict(self, x):
        return x @ self.coe

    def get_coe(self):
        a = self.coe
        b = self.bias
        return a, b

class NonlinearApprox:
    def __init__(self):
        self.coe = None