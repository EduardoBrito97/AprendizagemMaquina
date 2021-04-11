# Lista 4 - KMeans + Naive Bayes
* Descobre qual o melhor K para cada classe = ['TRUE', 'False'] de cada dataset utilizando os métodos 'Elbow' e 'Silhouette';
* Utiliza os melhores Ks para gerar x de treino e y de treino para um Naive Bayes Gaussiano
* Executa teste no GaussianNB treinado com os clusters encontrados pelo KMeans
* Datasets provenientes de [PROMISE DATASETS PAGE](http://promise.site.uottawa.ca/SERepository/datasets-page.html). Datasets escolhidos:
    * PC1/Software defect prediction
    * Class-level data for KC1 (Defective or Not)/Software defect prediction

Para gerar os resultados em txt train_and_test.py:
* dataset_index = 0 é o pc1.arff
* dataset_index = 1 é o kc1-class-level-defectiveornot.arff