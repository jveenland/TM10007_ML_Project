import pandas as pd
from glob import glob
import os
import numpy as np

folder = '/archive/mstarmans/Output/WORC_WORCPaper_Lipo_221205/Features'
output = '/home/mstarmans/TM10007Project/Lipo_radiomicFeatures.csv'
pinfo_file = '/home/mstarmans/Github/WORCDatabase/pinfo_Lipo.csv'


def load_label_txt(input_file):
    data = np.loadtxt(input_file, np.str)

    # Load and check the header
    header = data[0, :]

    # cut out the first header, only keep label header
    label_names = header[1::]

    # Patient IDs are stored in the first column
    patient_ID = data[1:, 0]

    # label status is stored in all remaining columns
    label_status = data[1:, 1:]
    label_status = label_status.astype(np.float)

    return label_names, patient_ID, label_status

def load_label_csv(input_file):
    """
    Load the patient IDs and label data from the label file

    Args:
        input_file (string): Path of the label file

    Returns:
        label_names (numpy array): Names of the different labels
        patient_ID (numpy array): IDs of patients for which label data is
         loaded
        label_status (numpy array): The status of the different labels
         for each patient
    """
    data = pd.read_csv(input_file, sep=None, header=0, engine='python')

    # Load and check the header
    header = data.keys()
    if header[0] != 'Patient':
        raise ae.WORCAssertionError('First column should be patient ID!')
    else:
        # cut out the first header, only keep label header
        label_names = header[1::]

    # Patient IDs are stored in the first column
    patient_ID = data['Patient'].values

    # label status is stored in all remaining columns
    label_status = data.values[:, 1:]
    label_status = label_status.astype(np.float)

    return label_names, patient_ID, label_status

# Load patient data
label_names, patient_ID, label_status = load_label_csv(pinfo_file)

# Loop over files
data = dict()
files = glob(os.path.join(folder, '*predict*.hdf5'))
files.sort()
totals = 0
for num, file in enumerate(files):
    # Only include if AD or CN
    index = None
    for pnum, p in enumerate(patient_ID):
        if p in file:
            index = pnum

    if index is not None:
        totals += 1
        print('Processing file ' + file)
        features = pd.read_hdf(file)
        feature_values = features.feature_values
        feature_labels = features.feature_labels
        if num == 0:
            # Initialize dict
            data = {k: list() for k in feature_labels}
            data['ID'] = list()
            data['label'] = list()

        for l, v in zip(feature_labels, feature_values):
            data[l].append(v)

        ID = os.path.basename(file)[12:-5]
        data['ID'].append(ID)

        label = label_status[index][0]
        if label == 0:
            label = 'T12'
        else:
            label = 'T34'

        data['label'].append(label)

# BUG: some keys occur multiple times, remove
for k in list(data.keys()):
    if len(data[k]) != totals:
        del data[k]

df = pd.DataFrame(data)
df.to_csv(output, index=False)
