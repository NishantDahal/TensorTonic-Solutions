import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0
    n = len(y)


    for i in range(epochs): 
        y_pred = np.dot(w, np.transpose(X)) + b
        dl_dw = 2/n*(np.dot(X.T, (y_pred - y)))
        dl_db = 2/n*np.sum(y_pred - y)

        w -= lr*dl_dw
        b -= lr*dl_db

    return w, b