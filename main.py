# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)


def titanicdata(csv_file, test = False):
    # Load dataset
    url = csv_file
    dataset = pd.read_csv(url, index_col=0)
    dataset = pd.read_csv(url)

    url = "test.csv"
    new_data = pd.read_csv(url, index_col=0)
    # print(new_data)

    dataset = dataset.drop('Name', 1)
    dataset = dataset.drop('Ticket', 1)
    dataset = dataset.drop('SibSp', 1)
    dataset = dataset.drop('Parch', 1)
    #dataset = dataset.drop('Fare', 1)
    dataset = dataset.drop('Cabin', 1)
    dataset = dataset.drop('Embarked', 1)

    if test is False:
        dataset = dataset[['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 'Survived']]
    else:
        dataset = dataset[['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare']]

    dataset['Sex'] = dataset['Sex'].replace('male', 0)
    dataset['Sex'] = dataset['Sex'].replace('female', 1)

    imp = SimpleImputer()
    dataset_numpy = imp.fit_transform(dataset)

    dataset_new = pd.DataFrame(dataset_numpy)
    dataset_new.columns = dataset.columns

    dataset = dataset_new
    #   dataset = dataset[np.isfinite(dataset['Age'])]

    return dataset


dataset = titanicdata('train.csv')


print(dataset.describe())
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:5]
Y = array[:,5]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7
scoring = 'accuracy'


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Make predictions on validation dataset
knn = LogisticRegression()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

to_test_dataset = titanicdata('test.csv', test = True)

predict = knn.predict(to_test_dataset)
to_submit = pd.DataFrame(predict)
to_submit = to_submit.rename(columns = {0: 'Survived'})
to_submit['PassengerId'] = to_test_dataset['PassengerId']
to_submit = to_submit.astype(int)
to_submit = to_submit[to_submit.columns[::-1]]
to_submit.to_csv('trial1_submit_LogisticRegression.csv', index=False)

