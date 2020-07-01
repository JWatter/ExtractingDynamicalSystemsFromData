import numpy as np

class LinearApprox:
	"""
	Linear approximation class, function calls similar to sklearn
	"""
    def __init__(self):
        self.coe = None
        self.bias = None
    
    def linear_approx(self, x, y):
        self.fit(x, y)
        return self.predict(x)
        
    def fit(self, x, y):
	"""
	Calculates the least squares solution between x and y
	using a numpy method
	"""
        self.coe = np.linalg.lstsq(x, y)[0]
        self.bias = np.linalg.lstsq(x, y)[1]

    def predict(self, x):
	"""
	Multiplies points x with calculated coefficients.
	Corresponds to one iteration / predicts new x.
	"""
        return x @ self.coe

    def get_coe(self):
	"""
	Returns coefficients
	"""
        a = self.coe
        b = self.bias
        return a, b

class NonlinearApprox:
	"""
	Similar to class LinearApprox
	"""
    def __init__(self, L, epsilon):
	"""
	L: number of (radial basis) functions
	epsilon: bandwidth of the radial basis functions
	"""
        self.L = L
        self.epsilon = epsilon
        self.center = None
        self.coe = None
        self.bias = None
        self.radial = None
    
    def radial_approx(self, x, y):
        self.fit(x, y)
        return self.predict(x)
        
    def fit(self, x, y):
	"""
	idx: picks random point in the dataset for the center of the basis function
	Calculates radial basis functions before performing linear last squares to
	get the unknown coefficients 
	"""
        idx = np.random.choice(x.shape[0], size=self.L, replace=False)
        self.center = x[idx].copy()
        radial = np.empty((x.shape[0], self.L))
        for l in range(self.L):
            norm = np.linalg.norm(x - self.center[l, :], axis=1)
            radial[:, l] = np.exp(-norm**2 / self.epsilon**2)
        self.radial = radial
        self.coe = np.linalg.lstsq(self.radial, y)[0]
        self.bias = np.linalg.lstsq(self.radial, y)[1]

    def predict(self, x):
        return self.radial @ self.coe
    
    def get_coe(self):
        a = self.coe
        b = self.bias
        return a, b
