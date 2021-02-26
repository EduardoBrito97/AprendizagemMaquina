import pandas as pd
from scipy.io import arff

datasets = ["pc1.arff", "cocomo81.arff"]
dataset_labels = [
    ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'lOCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount'],
    ['rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced', 'loc', 'actual']
]

number_of_attributes = [len(dataset_labels[0]), len(dataset_labels[1])]

def get_dataset(dataset_index):
    data = arff.loadarff('datasets/' + datasets[dataset_index])
    df = pd.DataFrame(data[0])
    return df