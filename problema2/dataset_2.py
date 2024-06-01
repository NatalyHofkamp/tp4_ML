
import pandas as pd
import numpy as np 

def get_standardize_data(filename):
    df = pd.read_csv(filename)
    df = df.drop(0)
    df = df.apply(pd.to_numeric, errors='coerce')
    data = df.iloc[:, 1:].values 
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    
    data_normalized = (data - mean) / std
    return data_normalized

def get_data2(filename):
    df = pd.read_csv(filename)
    df = df.drop(0)
    df = df.apply(pd.to_numeric, errors='coerce')
    data = df.iloc[:, 1:].values 
    return data