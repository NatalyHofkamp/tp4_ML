
import pandas as pd
import numpy as np 


def get_data(filename):
    # Load the dataset
    data = pd.read_csv(filename, header=None)
    data = data.iloc[:, 1:].values  
    
    # Standardize the dataset (zero mean, unit variance)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    data_normalized = (data - mean) / std
    
    return data_normalized