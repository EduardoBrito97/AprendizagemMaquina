import knn_no_weight
import knn_weight
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import k_fold_cross_validation as cv

algorithms = (cv.no_weight, cv.weighted)
ks = cv.ks

cv.train_and_get_reports(algorithms[0], 1, knn_no_weight)
cv.train_and_get_reports(algorithms[1], 1, knn_weight)
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

plt.figure()
sns.lineplot(x="K", y="Avg Accuracy", hue="Algorithm", palette=["green", "blue"], data=df)
plt.savefig('graphs/accuracy.png')