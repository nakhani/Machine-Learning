import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

class LLS:
    def __init__(self):
        self.w = None 

    def fit(self, X, y):
        
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        
        self.w = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ (X_with_bias.T @ y)
        return self.w

    def predict(self, X):
        
        if self.w is None:
            raise ValueError("Weights have not been calculated. Please call 'fit' first.")
        
      
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Invalid input matrix: X cannot have zero rows or columns.")
        
      
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        return np.dot(X_with_bias, self.w)
    
    def evaluate(self, Y_pred, Y_true):

        mae = mean_absolute_error(Y_true, Y_pred)
        mse = np.mean((Y_true - Y_pred) ** 2)
        rmse = np.sqrt(mse) 
        r2 = r2_score(Y_true, Y_pred)
        
        return mae, mse, rmse, r2
