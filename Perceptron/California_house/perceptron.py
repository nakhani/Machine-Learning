import numpy as np 

class Perceptron:
    def __init__(self, input_size, learning_rate_w=0.001, learning_rate_b=0.01, epochs=50):
        self.w = np.random.rand(input_size, 1)  
        self.b = np.random.rand(1, 1)         
        self.learning_rate_w = learning_rate_w
        self.learning_rate_b = learning_rate_b
        self.epochs = epochs
        self.loss_history = []                
    
    def fit(self, X_train, y_train):
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(X_train.shape[0]):
                x = X_train[i].reshape(-1, 1)  
                y = y_train[i]
                
                y_pred = np.dot(x.T, self.w) + self.b 
                
                error = y - y_pred  
                
                total_loss += error 
                

                self.w += error * x * self.learning_rate_w
                self.b += error * self.learning_rate_b
            

            self.loss_history.append(total_loss / X_train.shape[0])
            print(f"Epoch {epoch+1}: Loss = {total_loss / X_train.shape[0]}")
    
    def predict(self, X_test):
        predictions = []
        for i in range(X_test.shape[0]):
            x = X_test[i].reshape(-1, 1)  
            y_pred = np.dot(x.T, self.w) + self.b
            predictions.append(y_pred.flatten())
        return np.array(predictions)
