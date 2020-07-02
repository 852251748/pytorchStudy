from sklearn import datasets, preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    scaler = preprocessing.StandardScaler().fit(x)
    x_test = scaler.transform(x)
    krange = range(1, 31)
    kscore = []
    for n in krange:
        knn = neighbors.KNeighborsClassifier(n)
        knn.fit(x_test, y)

        scores = cross_val_score(knn, x_test, y, cv=5, scoring='accuracy')
        kscore.append(scores.mean())

    plt.figure()
    plt.plot(krange, kscore)
    plt.xlabel('Value of KNN')
    plt.show()
