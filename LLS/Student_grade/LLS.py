import numpy as np

class LLS:
    def __init__(self):
       
        self.w = None  

    def calculate_weights(self, X, y):
     
        
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        self.w = np.linalg.inv(X_with_bias.T * X_with_bias) * X_with_bias.T * y

    def predict(self, X):
    
        if self.w is None:
            raise ValueError("Weights have not been calculated. Please call 'calculate_weights' first.")
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        return X_with_bias * self.w
