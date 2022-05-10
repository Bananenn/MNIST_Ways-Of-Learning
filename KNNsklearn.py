import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


class myKNNsklearn:
    def __init__(self, k=5):
        self.k = k
        self.classifier = KNeighborsClassifier(n_neighbors=k)

    def learn(self):
        PATH = "mnist_train.csv"
        data = pd.read_csv(PATH).values
        y = np.array(data[:,0])
        self.X = np.array(data[:,1:])
        self.X = np.where(self.X == 0, 0, 1)
        self.classifier.fit(self.X,y)
        #plt.imshow(self.X[0].reshape(28,28))
        #plt.show()

    def predict(self, Xx):
        n = self.classifier.kneighbors(Xx)
        fig, ax = plt.subplots(1,self.k+1)
        ax = ax.ravel()
        ax[0].imshow(Xx.reshape((28,28)))
        ax[0].set_title("Original")
        ax[0].axis('off')
        for i,p in enumerate(ax[1:]):
            p.imshow(self.X[n[1][0][i]].reshape(28,28))
            p.set_title(f"nr: {i}")
            p.axis('off')
        plt.show()
        #plt.imshow(self.X[n[1][0]].reshape(28,28))
        #plt.show()
        return self.classifier.predict(Xx)

class mySVC:
    def __init__(self):
        self.clf = svm.SVC(kernel='rbf', C=100, gamma=0.001)

    def learn(self):
        PATH = "mnist_train.csv"
        data = pd.read_csv(PATH).values[:10000]
        y = np.array(data[:,0])
        X = np.array(data[:,1:])
        X = np.where(X == 0, 0, 1)
        self.clf.fit(X,y)
        """
        PATH = "mnist_test.csv"
        data = pd.read_csv(PATH).values[:500]
        y_test = np.array(data[:,0])
        X_test = np.array(data[:,1:])
        print(self.clf.score(X_test,y_test))
        print("SVC - Fit completed")
        """

    def predict(self, X):
        return self.clf.predict(X)

#obj = myKNNsklearn()
#obj.learn()