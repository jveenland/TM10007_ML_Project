import pandas as pd
import numpy as np
import os

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'ecg_data.csv'), index_col=0)
    data = pd.DataFrame(data)
    return data

def extract_data(input_data):
    ## Creating feature list for each patient (X) and truth list (Y) 
    # Convert dataframe to list for each row
    data_list = [row.tolist() for _, row in input_data.iterrows()]

    # Extract last row from data list as truth Y
    Y = []
    for row in data_list:
        Y.append(row[-1])    
    Y = np.array(Y)

    # Remove last column (truth) and create X
    for row in data_list:
        del row[-1]
    X = data_list
    X = np.array(X)
    return X, Y