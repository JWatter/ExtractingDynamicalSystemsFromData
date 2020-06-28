import numpy as np

class LinearApprox:
    def __init__(self):
        self.coe = None
        self.bias = None
    
    def linear_approx(self, x, y):
        self.fit(x, y)
        return self.predict(x)
        
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
    def __init__(self, L, epsilon):
        self.L = L
        self.epsilon = epsilon
        self.center = None
        self.coe = None
        self.bias = None
    
    def radial_approx(self, x, y):
        self.fit(x, y)
        return self.predict(x)
        
    def fit(self, x, y):
        idx = np.random.choice(x.shape[0], size=self.L, replace=False)
        self.center = x[idx].copy()
        radial = np.empty((x.shape[0], self.L))
        for l in range(self.L):
            norm = np.linalg.norm(x - self.center[l, :], axis=1)
            radial[:, l] = np.exp(-norm**2 / self.epsilon**2)
        self.coe = np.linalg.lstsq(radial, y)[0]
        self.bias = np.linalg.lstsq(radial, y)[1]

    def predict(self, x):
        radial = np.empty((x.shape[0], self.L))
        for l in range(self.L):
            norm = np.linalg.norm(x - self.center[l, :], axis=1)
            radial[:, l] = np.exp(-norm**2 / self.epsilon**2)
        return radial @ self.coe
    
    def get_coe(self):
        a = self.coe
        b = self.bias
        return a, b