import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn import preprocessing

datasets = ["pc1.arff", "kc1-class-level-defectiveornot.arff"]
number_of_attributes = [21, 94]

def get_dataset(dataset_index):
    data = arff.loadarff('datasets/' + datasets[dataset_index])
    df = pd.DataFrame(data[0])
    df = df.dropna(how='any',axis=0)
    df = df.sample(frac = 1)
    if dataset_index == 0:
        df['defects'] = df['defects'].astype(str)
    elif dataset_index == 1:
        df['DL'] = df['DL'].astype(str)
    return df

def get_1_class_instance(dataset_index, instance_val):
    real_value = ''
    if dataset_index == 0 and instance_val:
        real_value = 'b\'true\''
    elif dataset_index == 0 and not instance_val:
        real_value = 'b\'false\''
    elif dataset_index == 1 and instance_val:
        real_value = 'b\'_TRUE\''
    else:
        real_value = 'b\'FALSE\''

    df = get_dataset(dataset_index)

    df_num = df.select_dtypes(include=[np.number])
    df_norm = (df_num - df_num.mean()) / (df_num.max() - df_num.min() + 0.000001)
    df[df_norm.columns] = df_norm

    if dataset_index == 0:
        return df.loc[df['defects'] == real_value]
    else:
        return df.loc[df['DL'] == real_value]


def get_columns_and_class_column(dataset):
    columns = dataset.columns.values
    class_column = columns[-1]
    columns = columns[:-1]
    return columns, class_column

if __name__ == '__main__':
    df = get_1_class_instance(1, False)
    print(df.head())