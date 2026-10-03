import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    # pos - outer loop : seq_length, 1 .. 2 .. 3 
    # i - counter loop (sin - cosine) , d_m / 2 - 1 , 1.. 2... 3  
    # d_model : width
    # output : np.array(seq, dmodel)
    # pos : 1 , d_m : 0 , i : 0
    # sin 1/(10000)^(0)
    positional_values = np.zeros((seq_length, d_model))
    for pos in range(seq_length): 
        for i in range(d_model//2):
            expo = 2*i / d_model
            denom = 10_000**(2*i/d_model)
            positional_values[pos][2*i] = np.sin(pos/denom) 
            positional_values[pos][2*i+1] = np.cos(pos/denom)
    return positional_values