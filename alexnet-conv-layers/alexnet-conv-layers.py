import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    # YOUR CODE HERE
    kernel = 11
    stride = 4
    padding = 2
    image_d = image.shape #b, h, w, c
    output = np.floor((image_d[1]+2*padding-kernel)/stride)+ 1
    o_filter = 96
    return np.zeros((image_d[0], int(output), int(output), o_filter))