import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from dataset_reader import get_dataset, number_of_attributes
from train_and_test import train_and_test

kfold_num = 5
def gen_reports_and_statistics(dataset, dataset_target, dataset_name, dataset_index, reports, statistics):
    kfold = StratifiedKFold(n_splits=kfold_num, shuffle=True)
    fold_num = 0
    for train_index, test_index in kfold.split(dataset, dataset_target):
        train = dataset.iloc[train_index,:]
        x_train = train.iloc[:, :-1]
        y_train = train.iloc[:, number_of_attributes[dataset_index]]

        test = dataset.iloc[test_index,:]
        x_test = test.iloc[:, :-1]
        y_test = test.iloc[:, number_of_attributes[dataset_index]]

        tp, fp, tn, fn, y_pred, y_true = train_and_test(x_train, y_train, x_test, y_test)
        breakpoint()
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
        reports[dataset_name][fold_num] = report
        
        statistics[dataset_name][fold_num] = {}
        statistics[dataset_name][fold_num]['tp'] = tp
        statistics[dataset_name][fold_num]['fp'] = tp
        statistics[dataset_name][fold_num]['tn'] = tn
        statistics[dataset_name][fold_num]['fn'] = fn
        fold_num = fold_num + 1

def sum_reports(reports_sum, reports, dataset_name, fold):
    for param in reports[dataset_name][fold].keys():
        if type(reports[dataset_name][fold][param]) != type({}) and param not in reports_sum[dataset_name]:
            reports_sum[dataset_name][param] = reports[dataset_name][fold][param]
        elif type(reports[dataset_name][fold][param]) != type({}):
            reports_sum[dataset_name][param] = reports_sum[dataset_name][param] + reports[dataset_name][fold][param]
        elif param not in reports_sum[dataset_name]:
            reports_sum[dataset_name][param] = Counter(reports[dataset_name][fold][param])
        else:
            reports_sum[dataset_name][param] = reports_sum[dataset_name][param] + Counter(reports[dataset_name][fold][param])

def sum_statistics(statistics_sum, statistics, dataset_name, fold):
    statistics_sum[dataset_name]['tp'] = statistics_sum[dataset_name]['tp'] + statistics[dataset_name][fold]['tp']
    statistics_sum[dataset_name]['fp'] = statistics_sum[dataset_name]['fp'] + statistics[dataset_name][fold]['fp']
    statistics_sum[dataset_name]['tn'] = statistics_sum[dataset_name]['tn'] + statistics[dataset_name][fold]['tn']
    statistics_sum[dataset_name]['fn'] = statistics_sum[dataset_name]['fn'] + statistics[dataset_name][fold]['fn']

def process_reports(reports, statistics):
    reports_avg = {}
    statistics_avg = {}

    # Somando os valores para todos os scores avaliados
    for dataset_name in reports.keys():
        statistics_avg[dataset_name] = {}
        statistics_avg[dataset_name]['tp'] = 0
        statistics_avg[dataset_name]['fp'] = 0
        statistics_avg[dataset_name]['tn'] = 0
        statistics_avg[dataset_name]['fn'] = 0

        reports_avg[dataset_name] = {}
        for fold in reports[dataset_name].keys():
            sum_statistics(statistics_avg, statistics, dataset_name, fold)
            sum_reports(reports_avg, reports, dataset_name, fold)
    
    # Tirando as médias
    for dataset_name in reports_avg.keys():
        statistics_avg[dataset_name]['tp'] = statistics_avg[dataset_name]['tp'] / kfold_num
        statistics_avg[dataset_name]['fp'] = statistics_avg[dataset_name]['fp'] / kfold_num
        statistics_avg[dataset_name]['tn'] = statistics_avg[dataset_name]['tn'] / kfold_num
        statistics_avg[dataset_name]['fn'] = statistics_avg[dataset_name]['fn'] / kfold_num
        
        for param in reports_avg[dataset_name].keys():
            if type(reports_avg[dataset_name][param]) != type(Counter({})):
                reports_avg[dataset_name][param] = reports_avg[dataset_name][param] / kfold_num
            else:
                reports_avg[dataset_name][param] = dict(reports_avg[dataset_name][param])
                for score in reports_avg[dataset_name][param].keys():
                    reports_avg[dataset_name][param][score] = reports_avg[dataset_name][param][score] / kfold_num
            
    return reports_avg, statistics_avg

def gen_txt(reports, statistics):
    report_avg, statistics_avg = process_reports(reports, statistics)
    for dataset in report_avg.keys():
        folder_name = "results/" + dataset + "/" 
        report = report_avg[dataset]
        statistic = statistics_avg[dataset]

        file_name = folder_name + "results.txt"
        fo = open(file_name, "w")

        for param in report.keys():
            fo.write(param + ": " + str(report[param]))
            fo.write('\n')

        for param in statistic.keys():
            fo.write(param + ": " + str(statistic[param]))
            fo.write('\n')

        fo.close()

if __name__ == '__main__':
    global train_class_percentage
    statistics = {}
    statistics['pc1'] = {}
    statistics['kc1'] = {}

    reports = {}
    reports['pc1'] = {}
    reports['kc1'] = {}

    pc1 = get_dataset(0)
    pc1_target = pc1.loc[:, 'defects']

    kc1 = get_dataset(1)
    kc1_target = kc1.loc[:, 'DL']

    gen_reports_and_statistics(pc1, pc1_target, 'pc1', 0, reports, statistics)
    gen_reports_and_statistics(kc1, kc1_target, 'kc1', 1, reports, statistics)
    gen_txt(reports, statistics)