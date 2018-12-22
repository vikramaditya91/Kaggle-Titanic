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
    dataset = pd.read_csv(url)

    dataset = dataset.drop('Name', 1)
    dataset = dataset.drop('Ticket', 1)
    #dataset = dataset.drop('SibSp', 1)
    #dataset = dataset.drop('Parch', 1)
    #dataset = dataset.drop('Fare', 1)
    #dataset = dataset.drop('Cabin', 1)
    dataset = dataset.drop('Embarked', 1)

    if test is False:
        dataset = dataset[['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived']]
    else:
        dataset = dataset[['PassengerId', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]

    dataset['Sex'] = dataset['Sex'].replace('male', 0)
    dataset['Sex'] = dataset['Sex'].replace('female', 1)

    imp = SimpleImputer()
    dataset_numpy = imp.fit_transform(dataset)

    dataset_new = pd.DataFrame(dataset_numpy)
    dataset_new.columns = dataset.columns

    dataset = dataset_new
    #   dataset = dataset[np.isfinite(dataset['Age'])]

    return dataset


dataset = titanicdata('test.csv')


color_list = list(dataset['Survived'])
color_list = ['blue' if item==0 else 'red' for item in color_list]
dataset = titanicdata('train.csv')

dataset = titanicdata('train.csv', test=True)


scatter_matrix(dataset, color = color_list)


#scatter_matrix(dataset)
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

