import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from dataset_reader import get_dataset, number_of_attributes
from train_and_test import train_and_test

kfold_num = 5
kmeans_dict_key = 'kmeans+nb'
nb_dict_key = 'nb'
one_nn_dict_key = '1nn'

def sum_reports(reports_sum, reports, dataset_name, fold, alg):
    report_alg = reports[dataset_name][alg][fold]
    report_sum_alg = reports_sum[dataset_name][alg]
    
    for param in report_alg.keys():
        curr_param_val = report_alg[param]
        if type(curr_param_val) != type({}) and param not in reports_sum[dataset_name]:
            report_sum_alg[param] = curr_param_val
        elif type(report_alg[param]) != type({}):
            report_sum_alg[param] = report_sum_alg[param] + curr_param_val
        elif param not in report_sum_alg:
            report_sum_alg[param] = Counter(curr_param_val)
        else:
            report_sum_alg[param] = report_sum_alg[param] + Counter(curr_param_val)

def sum_statistics(statistics_sum, statistics, dataset_name, alg, fold):
    statistic_sum_alg = statistics_sum[dataset_name][alg]
    statistic_fold = statistics[dataset_name][alg][fold]
    statistic_sum_alg['tp'] = statistic_sum_alg['tp'] + statistic_fold['tp']
    statistic_sum_alg['fp'] = statistic_sum_alg['fp'] + statistic_fold['fp']
    statistic_sum_alg['tn'] = statistic_sum_alg['tn'] + statistic_fold['tn']
    statistic_sum_alg['fn'] = statistic_sum_alg['fn'] + statistic_fold['fn']

def process_reports(reports, statistics):
    reports_avg = {}
    statistics_avg = {}

    # Somando os valores para todos os scores avaliados
    for dataset_name in reports.keys():
        reports_avg[dataset_name] = {}
        statistics_avg[dataset_name] = {}

        for alg in reports[dataset_name].keys():
            reports_avg[dataset_name][alg] = {}

            statistics_avg[dataset_name][alg] = {}
            statistic_avg_alg = statistics_avg[dataset_name][alg]
            statistic_avg_alg['tp'] = 0
            statistic_avg_alg['fp'] = 0
            statistic_avg_alg['tn'] = 0
            statistic_avg_alg['fn'] = 0

            for fold in reports[dataset_name][alg].keys():
                sum_reports(reports_avg, reports, dataset_name, fold, alg)
                sum_statistics(statistics_avg, statistics, dataset_name, alg, fold)

    # Tirando as médias
    for dataset_name in reports_avg.keys():      
        for alg in reports_avg[dataset_name].keys():
            statistic_avg_alg = statistics_avg[dataset_name][alg]
            statistic_avg_alg['tp'] = statistic_avg_alg['tp'] / kfold_num
            statistic_avg_alg['fp'] = statistic_avg_alg['fp'] / kfold_num
            statistic_avg_alg['tn'] = statistic_avg_alg['tn'] / kfold_num
            statistic_avg_alg['fn'] = statistic_avg_alg['fn'] / kfold_num

            for param in reports_avg[dataset_name][alg].keys():
                update_param_val(reports_avg, dataset_name, alg, param)
            
    return reports_avg, statistics_avg

def update_param_val(reports_avg, dataset_name, alg, param):
    if type(reports_avg[dataset_name][alg][param]) != type(Counter({})):
        reports_avg[dataset_name][alg][param] = reports_avg[dataset_name][alg][param] / kfold_num
    else:
        reports_avg[dataset_name][alg][param] = dict(reports_avg[dataset_name][alg][param])
        for score in reports_avg[dataset_name][alg][param].keys():
            reports_avg[dataset_name][alg][param][score] = reports_avg[dataset_name][alg][param][score] / kfold_num

def gen_txt(reports, statistics):
    report_avg, statistics_avg = process_reports(reports, statistics)
    for dataset in report_avg.keys():
        for alg in report_avg[dataset].keys():
            folder_name = "results/" + dataset + "/" 
            report = report_avg[dataset][alg]
            statistic = statistics_avg[dataset][alg]

            file_name = folder_name + str(alg) + "_results.txt"
            fo = open(file_name, "w")

            for param in report.keys():
                fo.write(param + ": " + str(report[param]))
                fo.write('\n')

            for param in statistic.keys():
                fo.write(param + ": " + str(statistic[param]))
                fo.write('\n')

        fo.close()

def get_confusion_matrix_metrics(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TP, FP, TN, FN

def gen_reports_and_statistics(dataset, dataset_target, dataset_name, dataset_index, reports, statistics):
    kfold = StratifiedKFold(n_splits=kfold_num, shuffle=True)
    fold_num = 0
    
    reports[dataset_name][one_nn_dict_key] = {}
    reports[dataset_name][nb_dict_key] = {}
    reports[dataset_name][kmeans_dict_key] = {}
    
    statistics[dataset_name][one_nn_dict_key] = {}
    statistics[dataset_name][nb_dict_key] = {}
    statistics[dataset_name][kmeans_dict_key] = {}

    for train_index, test_index in kfold.split(dataset, dataset_target):
        print('Training dataset ' + dataset_name + ', Fold ' + str(fold_num + 1) + ' out of ' + str(kfold_num) + '...')
        
        train = dataset.iloc[train_index,:]
        x_train = train.iloc[:, :-1]
        y_train = train.iloc[:, number_of_attributes[dataset_index]:]

        test = dataset.iloc[test_index,:]
        x_test = test.iloc[:, :-1]
        y_test = test.iloc[:, number_of_attributes[dataset_index]:]

        # Treinando e pegando métricas do 1NN
        classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform')
        classifier.fit(x_train, y_train.values.ravel(order='C'))
        
        y_pred_1nn = classifier.predict(x_test)
        report_1nn = classification_report(y_test, y_pred_1nn, output_dict=True, zero_division=1)
        reports[dataset_name][one_nn_dict_key][fold_num] = report_1nn
        
        tp, fp, tn, fn = get_confusion_matrix_metrics(y_test, y_pred_1nn)
        statistics[dataset_name][one_nn_dict_key][fold_num] = {}
        statistics[dataset_name][one_nn_dict_key][fold_num]['tp'] = tp
        statistics[dataset_name][one_nn_dict_key][fold_num]['fp'] = tp
        statistics[dataset_name][one_nn_dict_key][fold_num]['tn'] = tn
        statistics[dataset_name][one_nn_dict_key][fold_num]['fn'] = fn

        # Treinando e pegando métricas do Naive Bayes puro
        gnb = GaussianNB()
        gnb.fit(x_train, y_train.values.ravel(order='C'))

        y_pred_nb = gnb.predict(x_test)
        report_nb = classification_report(y_test, y_pred_nb, output_dict=True, zero_division=1)
        reports[dataset_name][nb_dict_key][fold_num] = report_nb
        
        tp, fp, tn, fn = get_confusion_matrix_metrics(y_test, y_pred_nb)
        statistics[dataset_name][nb_dict_key][fold_num] = {}
        statistics[dataset_name][nb_dict_key][fold_num]['tp'] = tp
        statistics[dataset_name][nb_dict_key][fold_num]['fp'] = tp
        statistics[dataset_name][nb_dict_key][fold_num]['tn'] = tn
        statistics[dataset_name][nb_dict_key][fold_num]['fn'] = fn

        # Treinando e pegando métricas do algoritmo sugerido
        y_pred, y_true = train_and_test(x_train, y_train, x_test, y_test)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
        reports[dataset_name][kmeans_dict_key][fold_num] = report

        tp, fp, tn, fn = get_confusion_matrix_metrics(y_true, y_pred)
        statistics[dataset_name][kmeans_dict_key][fold_num] = {}
        statistics[dataset_name][kmeans_dict_key][fold_num]['tp'] = tp
        statistics[dataset_name][kmeans_dict_key][fold_num]['fp'] = tp
        statistics[dataset_name][kmeans_dict_key][fold_num]['tn'] = tn
        statistics[dataset_name][kmeans_dict_key][fold_num]['fn'] = fn

        fold_num = fold_num + 1
    print('Finished ' + dataset_name + '.')

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