# Lista 2 - LVQ
* Avaliação usando 10-Fold Cross-Validation.
* Aplicação dos algoritmos, todos com distância euclidiana como função avaliativa:
    * k-NN e distância euclidiana;
* Variar o parâmetro k em 1, 3 e 5.
* Gerando protótipos com LVQ1, LVQ2.1 e LVQ3
* Datasets provenientes de [PROMISE DATASETS PAGE](http://promise.site.uottawa.ca/SERepository/datasets-page.html). Datasets escolhidos:
    * PC1/Software defect prediction
    * Class-level data for KC1 (Defective or Not)/Software defect prediction

Para gerar os resultados em txt e gráficos, rodar plot_results.py. A código que roda os diferentes algoritmos e os avalia está em k_fold_cross_validation.train_and_get_reports:
* dataset_index = 0 é o pc1.arff
* dataset_index = 1 é o kc1-class-level-defectiveornot.arff