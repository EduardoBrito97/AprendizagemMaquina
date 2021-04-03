# Lista 3 - One Class classification
* Propor um algoritmo que consiga predizer caso uma instância pertence ou não à mesma classe que as instâncias de treinamento
* Algoritmo proposto:
    * Tentar eliminar ruídos da seguinte forma:
        * Caso exista ao menos 3 instâncias de mesma classe à uma distância X da instância analisada, ela é usada como treinamento
        * Caso não exista, ela é removida do conjunto de treinamento
    * Criar uma região de aceitação: 
        * Ao analisar uma nova instância, caso ela esteja à uma distância Y < X, ela é dita como da classe treinada
        * Caso não esteja, é dita de outra classe
    * Tanto o dataset quanto a instância nova analisada serão normalizadas
* Datasets provenientes de [PROMISE DATASETS PAGE](http://promise.site.uottawa.ca/SERepository/datasets-page.html). Datasets escolhidos:
    * PC1/Software defect prediction
    * Class-level data for KC1 (Defective or Not)/Software defect prediction

Para gerar os resultados em txt train_and_test.py:
* dataset_index = 0 é o pc1.arff
* dataset_index = 1 é o kc1-class-level-defectiveornot.arff