import numpy as np
import pickle

class MLP:
    def __init__(self, input_size, layer_1, layer_2, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.w1 = np.random.randn(input_size, layer_1)
        self.w2 = np.random.randn(layer_1, layer_2)
        self.w3 = np.random.randn(layer_2, output_size)
        self.B1 = np.random.randn(1, layer_1)
        self.B2 = np.random.randn(1, layer_2)
        self.B3 = np.random.randn(1, output_size)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def softmax(self, X):
        X_max = np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X - X_max)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def root_mean_squared_error(self, Y_gt, Y_pred):
        return np.sqrt(np.mean((Y_gt - Y_pred) ** 2))

    def fit(self, X_train, Y_train, epochs):
        for epoch in range(epochs):
            Y_pred = []
            for x, y in zip(X_train, Y_train):
                x = x.reshape(-1, 1)

                # Forward propagation
                out1 = self.sigmoid(x.T @ self.w1 + self.B1)
                out2 = self.sigmoid(out1 @ self.w2 + self.B2)
                out3 = self.softmax(out2 @ self.w3 + self.B3)
                y_pred = out3
                Y_pred.append(y_pred)

                # Backpropagation
                error_out = -2 * (y - y_pred)
                grad_B3 = error_out
                grad_w3 = out2.T @ error_out

                error_l2 = error_out @ self.w3.T * out2 * (1 - out2)
                grad_B2 = error_l2
                grad_w2 = out1.T @ error_l2

                error_l1 = error_l2 @ self.w2.T * out1 * (1 - out1)
                grad_B1 = error_l1
                grad_w1 = x @ error_l1

                # Update parameters
                self.w1 -= self.learning_rate * grad_w1
                self.B1 -= self.learning_rate * grad_B1
                self.w2 -= self.learning_rate * grad_w2
                self.B2 -= self.learning_rate * grad_B2
                self.w3 -= self.learning_rate * grad_w3
                self.B3 -= self.learning_rate * grad_B3

            # Epoch logging
            Y_pred = np.array(Y_pred).reshape(-1, 10)
            loss = self.root_mean_squared_error(Y_train, Y_pred)
            accuracy = np.sum(np.argmax(Y_train, axis=1) == np.argmax(Y_pred, axis=1)) / len(Y_train)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}, Accuracy: {accuracy:.2f}")

    def predict(self, X):
        predictions = []
        for x in X:
            x = x.reshape(-1, 1)
            out1 = self.sigmoid(x.T @ self.w1 + self.B1)
            out2 = self.sigmoid(out1 @ self.w2 + self.B2)
            out3 = self.softmax(out2 @ self.w3 + self.B3)
            predictions.append(out3)
        return np.array(predictions)
    

    def evaluate(self, X, Y):
        Y_pred = self.predict(X).reshape(-1, 10)
        loss = self.root_mean_squared_error(Y, Y_pred)
        accuracy = np.sum(np.argmax(Y, axis=1) == np.argmax(Y_pred, axis=1)) / len(Y)
        return loss, accuracy

    def save(self, filename="mlp_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}.")
