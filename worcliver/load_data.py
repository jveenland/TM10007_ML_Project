import pandas as pd
import os


def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Liver_radiomicFeatures.csv'), index_col=0)

    return data


def load_Ft_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    Ft_usable = pd.read_csv(os.path.join(this_directory, 'Ft_set.csv'), index_col=0)

    return Ft_usable


def load_D_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    D_usable = pd.read_csv(os.path.join(this_directory, 'D_set.csv'), index_col=0)

    return D_usable
