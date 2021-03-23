import knn_no_weight
import os
import lvq1, lvq2, lvq3
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import k_fold_cross_validation as cv
from sklearn.metrics import roc_curve
from dataset_reader import datasets
from sklearn.metrics import roc_curve

ks = cv.ks
algorithms = ['lvq1', 'lvq2', 'lvq3']

def train_and_plot_graphs(dataset_index):
        cv.train_and_test_on_dataset_and_save_results(dataset_index)
        reports_avg = cv.process_reports()

        algorithms_order = []
        ks_order = []
        accuracies = []
        train_times = []
        test_times = []
        for algorithm in algorithms:
                for k in ks:
                        accuracies.append(reports_avg[algorithm][k]['accuracy'])
                        train_times.append(reports_avg[algorithm][k]['train_time_in_ms'])
                        test_times.append(reports_avg[algorithm][k]['test_time_in_ms'])
                        algorithms_order.append(algorithm)
                        ks_order.append(k)

        data = {
                'K' : ks_order,
                'Algorithm' : algorithms_order,
                'Avg Accuracy' : accuracies,
                'Train time in ms' : train_times,
                'Test time in ms' : test_times
        }

        df = pd.DataFrame(data, columns = ['K', 'Algorithm', 'Avg Accuracy', 'Train time in ms', 'Test time in ms'])
        
        dataset_name = datasets[dataset_index]
        folder_name = "graphs/" + dataset_name + "/"
        if os.path.exists(folder_name) == False:
                os.mkdir(folder_name, mode = 0o666)

        plt.figure()
        sns.lineplot(x="K", y="Avg Accuracy", hue="Algorithm", palette=["green", "blue", "black"], data=df)
        plt.savefig(folder_name + "accuracy.png")

        plt.figure()
        sns.lineplot(x="K", y="Train time in ms", hue="Algorithm", palette=["green", "blue", "black"], data=df)
        plt.savefig(folder_name + "train_time.png")

        plt.figure()
        sns.lineplot(x="K", y="Test time in ms", hue="Algorithm", palette=["green", "blue", "black"], data=df)
        plt.savefig(folder_name + "test_time.png")
        return reports_avg

def plot_roc_curve(reports_avg, dataset_index):
        dataset_name = datasets[dataset_index]
        folder_name = "graphs/" + dataset_name + "/"
        if os.path.exists(folder_name) == False:
                os.mkdir(folder_name, mode = 0o666)

        true_positives = {}
        for algorithm in algorithms:
                true_positives[algorithm] = {}
                for k in ks:
                        true_positives[algorithm][k] = float(reports_avg[algorithm][k][knn_no_weight.true_positive_precision])

        k_colors = ['b', 'g', 'r']
        for algorithm in algorithms:
                plt.figure()
                i = 0
                for k in ks:
                        true_positive = true_positives[algorithm][k]              
                        plt.plot( 1 - true_positive, true_positive, 'o', label='K = ' + str(k), color=k_colors[i])
                        i = i + 1
                plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve - ' + str(algorithm))
                plt.legend()
                axis = plt.gca()
                axis.set_xlim([0.0,1.0])
                axis.set_ylim([0.0,1.0])
                plt.savefig(folder_name + "roc_curve_" + str(algorithm) + ".png")


if __name__ == "__main__":
        reports_avg = train_and_plot_graphs(0)
        plot_roc_curve(reports_avg, 0)
        reports_avg = train_and_plot_graphs(1)
        plot_roc_curve(reports_avg, 1)