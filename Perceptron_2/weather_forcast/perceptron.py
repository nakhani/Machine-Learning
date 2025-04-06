import numpy as np
from tqdm import tqdm

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=50, activation="linear"):
        self.w = np.random.rand(input_size, 1)  
        self.b = np.random.rand(1, 1)        
        self.activation = activation          
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_losses = []           
        self.train_accuracies = []            
        self.validation_losses = []            
        self.validation_accuracies = []        

    def activation_function(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function. Choose 'linear', 'sigmoid', 'relu', or 'tanh'.")

    def fit(self, X_train, y_train, X_validation, y_validation):
        for epoch in tqdm(range(self.epochs)):
            train_loss = []
            train_accuracy = []

            # Training 
            for x, y in zip(X_train, y_train):
                x = x.reshape(-1, 1) 
                y_pred = np.dot(x.T, self.w) + self.b
                y_pred_activated = self.activation_function(y_pred)


                error = y - y_pred_activated


                self.w += self.learning_rate * error * x
                self.b += self.learning_rate * error


                train_loss.append(np.mean(np.abs(error)))  
                train_accuracy.append(self.accuracy(y, y_pred_activated))


            val_loss, val_accuracy = self.evaluate(X_validation, y_validation)

            self.train_losses.append(np.mean(train_loss))
            self.train_accuracies.append(np.mean(train_accuracy))
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_accuracy)


            print(f"Epoch {epoch+1}:")
            print(f"Train Loss: {self.train_losses[-1]}, Train Accuracy: {self.train_accuracies[-1]}")
            print(f"Validation Loss: {self.validation_losses[-1]}, Validation Accuracy: {self.validation_accuracies[-1]}")


        self.train_losses = np.array(self.train_losses)
        self.train_accuracies = np.array(self.train_accuracies)
        self.validation_losses = np.array(self.validation_losses)
        self.validation_accuracies = np.array(self.validation_accuracies)

    def predict(self, X):
        Y_pred = []
        for x in X:
            x = x.reshape(-1, 1)
            y_pred = np.dot(x.T, self.w) + self.b
            y_pred_activated = self.activation_function(y_pred)
            Y_pred.append(y_pred_activated)
        return np.array(Y_pred)

    def accuracy(self, y_true, y_pred):
        y_pred = np.where(y_pred > 0.5, 1, 0)  
        return np.mean(y_pred == y_true)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        error = y - predictions
        loss = np.mean(np.abs(error))  
        accuracy = self.accuracy(y, predictions)
        return loss, accuracy

    def save(self, directory="./"):
        np.save(f"{directory}/weights.npy", self.w)
        np.save(f"{directory}/bias.npy", self.b)