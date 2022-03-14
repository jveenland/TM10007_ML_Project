import pandas as pd
import os


def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'TCGA_GBM_radiomicFeatures.csv'), index_col=0)
    data2 = pd.read_csv(os.path.join(this_directory, 'TCGA_LGG_radiomicFeatures.csv'), index_col=0)

    data['label'] = 'GBM'
    data2['label'] = 'LGG'
    data = data.append(data2)
    data = data.drop(columns=['Date'])

    return data
