from sklearn import datasets, preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    knn = neighbors.KNeighborsClassifier(n_neighbors=6)
    knn.fit(x_train, y_train)

    acores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    print(acores)
    print(acores.mean())

    y_pred = knn.predict(x_test)

    print(accuracy_score(y_pred, y_test))
