import numpy as np

class LinearRegression:
    def __init__(self, X, Y, fit_intercept=True, solver='OLS', lambd=0):
        self.X = np.copy(X)
        self.N, self.p = self.X.shape
        if fit_intercept:
            self.X = np.hstack((
                np.ones((self.N,1)), self.X
            ))
        self.Y = np.copy(Y)
        self.fit_in = fit_intercept
        self.solver = solver
        self.lambd  = lambd
        self.beta   = None

    def fit(self):
        if self.solver == 'OLS':
            self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        
        if self.solver == 'Ridge':
            self.beta = np.linalg.inv(self.X.T @ self.X + self.lambd * np.eye(self.p)) @ self.X.T @ self.Y

    def predict(self, X):
        Xt  = np.copy(X)
        N,p = X.shape
        if self.fit_in:
            Xt = np.hstack((
                np.ones((N,1)), Xt
            ))
        return Xt @ self.beta

class PolynomialRegression:
    def __init__(self, X, Y, q=1):
        self.X = np.copy(X)
        self.N, self.p = self.X.shape
        self.q = q
        for i in range(q):
            if i == 0:
                x = np.ones((self.N,1)) # intercept term
            else:
                x = X**(i+1) # higher order terms
            self.X = np.concatenate((
                self.X, x
            ), axis=1)
        self.Y = np.copy(Y)
        self.beta = None

    def fit(self):
        self.beta = np.linalg.pinv(self.X) @ self.Y

    def predict(self, X):
        Xt = np.copy(X)
        if len(X.shape)>2:
            z,k,j = X.shape
            for i in range(self.q):
                if i == 0:
                    x = np.ones((z,k,1))
                else:
                    x = (X)**(i+1)
                Xt = np.concatenate((
                    Xt, x
                ),axis=2)
            return Xt@self.beta
        else:
            N,p = X.shape
            for i in range(self.q):
                if i == 0:
                    x = np.ones((N,1))
                else:
                    x = (X)**(i+1)
                Xt = np.concatenate((
                    Xt, x
                ),axis=1)
            return Xt@self.beta