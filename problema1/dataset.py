import pandas as pd
import numpy as np 

def get_data(filename):
    df = pd.read_csv(filename)
    numeric_df = df.select_dtypes(include=[np.number])
    data = numeric_df.values
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    standardized_data = (data - mean) / std
    return standardized_data

