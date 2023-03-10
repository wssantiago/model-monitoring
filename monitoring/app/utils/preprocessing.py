import pandas as pd
import numpy as np


def data_preprocessing(dataframe):

    dataframe.fillna(value=np.nan, inplace=True)

    percent = dataframe.isnull().mean()
    columns_to_drop = percent[percent > 0.75].index
    dataframe = dataframe.drop(columns=columns_to_drop, axis=1)
    dataframe = dataframe.drop(
        columns=['VAR147', 'VAR148', 'VAR149', 'ID'], axis=1)

    X_ = dataframe
    if 'TARGET' in dataframe.columns:
        X_ = dataframe.loc[:, dataframe.columns != 'TARGET']

    return [dataframe, X_]
