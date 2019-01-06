import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


def sibling_parent_to_family(df):
    df['FSize'] = df['SibSp'] + df['Parch'] + 1
    df = df.drop('Parch', 1)
    df = df.drop('SibSp', 1)
    return df


def cabin_to_alphabet(df):
    df['Cabin'] = df['Cabin'].str[0]
    return df


def drop_stupid_shit(df):
    df = df.drop('Name', 1)
    df = df.drop('Ticket', 1)
    df = df.drop('PassengerId', 1)
    return df

def sex_to_int(df):
    df['Sex'] = df['Sex'].replace('male', 0)
    df['Sex'] = df['Sex'].replace('female', 1)
    return df


def get_color(df):
    color_list = list(df['Survived'])
    color_list = ['blue' if item==0 else 'red' for item in color_list]
    return color_list

def fill_in_empty_ages(df):
    men_df = df.loc[df['Sex'] == 0]
    men_avg_age = men_df['Age'].mean()

    fem_df = df.loc[df['Sex'] == 1]
    fem_avg_age = fem_df['Age'].mean()

    df['Age'] = df.apply(
        lambda row: men_avg_age if np.isnan(row['Age']) and row['Sex']==0
        else fem_avg_age if row['Sex']==1 else row['Age'],
        axis=1
    )
    return df

def set_missing_embark(df, embark):
    df['Embarked'] = df['Embarked'].fillna(embark)
    return df

def plot_scatter_matrix(df):
    color_list = get_color(df)
    to_disp = df[['Pclass', 'Sex', 'Fare', 'FSize', 'Embarked']]
    #scatter_matrix(to_disp, color=color_list)
    plt.show()


def embark_to_int(df):
    df['Embarked'] = df['Embarked'].replace('S', 0)
    df['Embarked'] = df['Embarked'].replace('C', 1)
    df['Embarked'] = df['Embarked'].replace('Q', 2)
    return df


def embardked_vs_fare(df):
    embarked_list = list(df['Embarked'])
    for item in embarked_list:
        embarked_list.append()
    return color_list


