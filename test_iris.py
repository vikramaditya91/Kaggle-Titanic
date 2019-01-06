import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm, metrics

iris = datasets.load_iris()
print(type(iris))
print(iris.data.shape)
print(iris.target.shape)
#print(iris.target_names)
#print(iris.DESCR)
#print(iris.target.shape)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf)
print(clf.score(X_test, y_test))


clf =  svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
print(n_samples)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
print(cross_val_score(clf, iris.data, iris.target, cv=cv))