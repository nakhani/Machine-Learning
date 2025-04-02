import numpy as np

def train_test_split(X, Y, test_size=0.2, random_state=None):
 
    X = np.array(X)
    Y = np.array(Y)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    X = X[indices]
    Y = Y[indices]
    
    split_index = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    
    return X_train, X_test, Y_train, Y_test

