import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class myKNNsklearn:
    def __init__(self):
        self.classifier = KNeighborsClassifier(n_neighbors=5)

    def learn(self):
        PATH = "mnist_train.csv"
        data = pd.read_csv(PATH).values[:40000]
        y = np.array(data[:,0])
        self.X = np.array(data[:,1:])
        self.X = np.where(self.X == 0, 0, 1)
        self.classifier.fit(self.X,y)
        print("Knn - Fit completed")
        """
        for i,dig in enumerate(self.X[0]):
            if i%28 == 0:
                print("")
            print(dig, end="")
        """
        #plt.imshow(self.X[0].reshape(28,28))
        #plt.show()

    def predict(self, Xx):
        n = self.classifier.kneighbors(Xx)
        fig, ax = plt.subplots(1,5)
        ax = ax.ravel()
        for i,p in enumerate(ax):
            p.imshow(self.X[n[1][0][i]].reshape(28,28))
        plt.show()
        #plt.imshow(self.X[n[1][0]].reshape(28,28))
        #plt.show()
        return self.classifier.predict(Xx)

class mySVC:
    def __init__(self):
        self.clf = svm.SVC(kernel='rbf', C=5, gamma=0.0007)

    def learn(self):
        PATH = "mnist_train.csv"
        data = pd.read_csv(PATH).values[:10000]
        y = np.array(data[:,0])
        X = np.array(data[:,1:])
        #X = np.where(X == 0, 0, 1)
        plt.imshow(X[5].reshape((28,28)))
        plt.show()
        self.clf.fit(X,y)
        print("SVC - Fit completed")

    def predict(self, X):
        return self.clf.predict(X)

#obj = myKNNsklearn()
#obj.learn()