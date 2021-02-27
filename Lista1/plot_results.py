import knn_no_weight
import knn_weight
import os
import adaptative_knn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import k_fold_cross_validation as cv
from dataset_reader import datasets

algorithms = (cv.no_weight, cv.weighted, cv.adaptative)
ks = cv.ks

def train_and_plot_graphs(dataset_index):
        cv.train_and_get_reports(algorithms[0], dataset_index, knn_no_weight)
        cv.train_and_get_reports(algorithms[1], dataset_index, knn_weight)
        cv.train_and_get_reports(algorithms[2], dataset_index, adaptative_knn)
        reports_avg = cv.process_reports()

        algorithms_order = []
        ks_order = []
        accuracies = []
        for algorithm in algorithms:
                for k in ks:
                        accuracies.append(reports_avg[algorithm][k]['accuracy'])
                        algorithms_order.append(algorithm)
                        ks_order.append(k)

        data = {
                'K' : ks_order,
                'Algorithm' : algorithms_order,
                'Avg Accuracy' : accuracies
        }

        df = pd.DataFrame(data, columns = ['K', 'Algorithm', 'Avg Accuracy'])
        
        dataset_name = datasets[dataset_index]
        os.mkdir("graphs/" + dataset_name + "/", mode = 0o666)

        plt.figure()
        sns.lineplot(x="K", y="Avg Accuracy", hue="Algorithm", palette=["green", "blue", "black"], data=df)
        plt.savefig("graphs/" + dataset_name + "/accuracy.png")

if __name__ == "__main__":
        train_and_plot_graphs(0)
        train_and_plot_graphs(1)