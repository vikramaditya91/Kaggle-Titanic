import helpers
import pandas as pd
import warnings
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)


''' Getting training data setup correctly'''

train_data = pd.read_csv('train.csv')

train_data = helpers.sibling_parent_to_family(train_data)

train_data = helpers.cabin_to_alphabet(train_data)

train_data = helpers.drop_stupid_shit(train_data)

train_data = helpers.sex_to_int(train_data)

color_list = helpers.get_color(train_data)

print(train_data.to_string())
to_disp =  train_data[['Pclass', 'Sex', 'Fare', 'FSize']]

scatter_matrix(to_disp, color = color_list)
plt.show()




