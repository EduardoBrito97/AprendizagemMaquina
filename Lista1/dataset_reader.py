import pandas as pd
from scipy.io import arff

datasets = ["pc1.arff", "kc1-class-level-defectiveornot.arff"]
number_of_attributes = [21, 94]

def get_dataset(dataset_index):
    data = arff.loadarff('datasets/' + datasets[dataset_index])
    df = pd.DataFrame(data[0])
    if dataset_index == 0:
        df['defects'] = df['defects'].astype(str)
    elif dataset_index == 1:
        df['DL'] = df['DL'].astype(str)
    return df