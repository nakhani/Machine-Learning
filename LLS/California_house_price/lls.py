import numpy as np

class LLS:
    def __init__(self):
        self.w = None  

    def calculate_weights(self, X, y):
        
        X_with_bias = np.c_[np.ones(X.shape[0]), X]  
        
        #w = (X^T * X)^(-1) * X^T * y
        self.w = np.matmul(np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)), np.matmul(X_with_bias.T, y))
        return self.w

    def predict(self, X):
        if self.w is None:
            raise ValueError("Weights have not been calculated. Please call 'calculate_weights' first.")
        
      
        X_with_bias = np.c_[np.ones(X.shape[0]), X]  
        
        
        return np.dot(X_with_bias, self.w)
