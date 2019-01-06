import helpers
import pandas as pd
import warnings

from os import path as op
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


warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)


''' Getting training data setup correctly'''
csv_dir = op.join(op.dirname(__file__), u'all_csv_files')
train_data = pd.read_csv(op.join(csv_dir, 'train.csv'))
train_data = helpers.sibling_parent_to_family(train_data)
train_data = helpers.cabin_to_alphabet(train_data)
train_data = helpers.drop_stupid_shit(train_data)
train_data = helpers.sex_to_int(train_data)
train_data = helpers.fill_in_empty_ages(train_data)
train_data = helpers.set_missing_embark(train_data, 'C')
train_data = helpers.embark_to_int(train_data)
#helpers.plot_scatter_matrix(train_data)

array_data = train_data.drop(['Survived', 'Cabin'], axis=1)
target_data = train_data['Survived'].values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(array_data, target_data, test_size=0.4, random_state=0)

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
    kfold = model_selection.KFold(n_splits=10, random_state=0)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
