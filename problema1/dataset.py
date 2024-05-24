
import pandas as pd

def get_data(filename):
    data = pd.read_csv(filename)
    return data.iloc[:, 1:].values  