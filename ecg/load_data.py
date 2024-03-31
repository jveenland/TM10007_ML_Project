import pandas as pd
import numpy as np
import os

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'ecg_data.csv'), index_col=0)
    data = pd.DataFrame(data)
    return data

def extract_data(input_data):
    # Create variable Y 
    Y = input_data['label']

    # Create variable X
    X = input_data.drop('label', axis=1)
    return X, Y