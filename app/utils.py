import numpy as np

def preprocess_input(input_list, scaler):
    array = np.array(input_list).reshape(1, -1)
    scaled = scaler.transform(array)
    return scaled