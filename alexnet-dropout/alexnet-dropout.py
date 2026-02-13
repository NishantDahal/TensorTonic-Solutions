import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """Apply dropout to input."""
    # YOUR CODE HERE
    m = np.random.binomial(n=1, p=1-p, size=x.shape)
    if training == True : 
        scaling = 1 / (1-p)
        masking = x * m
        return (scaling * masking)
    else :
        return (x)