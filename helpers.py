import pandas as pd

def sibling_parent_to_family(panda_dataframe):
    panda_dataframe['FSize'] = panda_dataframe['SibSp'] + panda_dataframe['Parch'] + 1
    panda_dataframe = panda_dataframe.drop('Parch', 1)
    panda_dataframe = panda_dataframe.drop('SibSp', 1)
    return panda_dataframe


def cabin_to_alphabet(panda_dataframe):
    panda_dataframe['Cabin'] = panda_dataframe['Cabin'].str[0]
    return panda_dataframe


def drop_stupid_shit(panda_dataframe):
    panda_dataframe = panda_dataframe.drop('Name', 1)
    panda_dataframe = panda_dataframe.drop('Ticket', 1)
    panda_dataframe = panda_dataframe.drop('PassengerId', 1)

    return panda_dataframe


def sex_to_int(panda_dataframe):
    panda_dataframe['Sex'] = panda_dataframe['Sex'].replace('male', 0)
    panda_dataframe['Sex'] = panda_dataframe['Sex'].replace('female', 1)
    return panda_dataframe


def get_color(panda_dataframe):
    color_list = list(panda_dataframe['Survived'])
    color_list = ['blue' if item==0 else 'red' for item in color_list]
    return color_list
