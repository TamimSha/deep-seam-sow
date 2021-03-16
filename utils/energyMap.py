from numpy.core.numeric import ones_like
from scipy.ndimage.filters import convolve
import numpy as np

def getEnergyMap(input):
    filter_x = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    filter_y = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]
    grey = np.dot(input[...,:3], [0.299, 0.587, 0.144]).astype(np.float32)/255.0
    # grey = np.dot(input[...,:3], [0.333, 0.334, 0.333]).astype(np.float32)/255.0
    filtered_x = convolve(grey, filter_x)
    filtered_y = convolve(grey, filter_y)
    # energy = np.math.sqrt(np.math.pow(filtered_x, 2) + np.math.pow(filtered_y, 2))
    energy = (filtered_x ** 2 + filtered_y ** 2) ** 0.5
    memorization = np.zeros_like(energy)
    memorization[:, :] = energy[:, :]
    minimum = 0
    maximum = energy.shape[1]
    for row in range(energy.shape[0] - 2, -1, -1):
        for col in range(minimum, maximum):
            left_col = max(col-1, minimum)
            right_col = min(col+2, maximum)
            value = np.min(memorization[row+1, left_col:right_col])
            memorization[row, col] += value
    return grey, energy, memorization