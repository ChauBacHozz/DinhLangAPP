import numpy as np
import pickle
from scipy.signal import savgol_filter
from sklearn.preprocessing import FunctionTransformer

# SNV function
def snv(input_data):
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data

# Feature selection function
def features_select(x):
    with open('file.pkl', 'rb') as file:
        params = pickle.load(file)
    sorted_ind = params['sorted_ind']
    wav = params['wav']
    return x[:, sorted_ind][:, wav:]
