import pandas as pd
import numpy as np


# Receives a DataFrame and drops columns with percentage null values
# greater than 75%. The remaining columns to drop are infered
# irrelevant and not considered in the model.pkl pipeline.
def data_preprocessing(dataframe):

    dataframe.fillna(value=np.nan, inplace=True)

    percent = dataframe.isnull().mean()
    columns_to_drop = percent[percent > 0.75].index
    dataframe = dataframe.drop(columns=columns_to_drop, axis=1)
    dataframe = dataframe.drop(
        columns=['VAR147', 'VAR148', 'VAR149', 'ID'], axis=1)

    # The X matrix is returned in the correct shape for entering
    # the pipeline transformation and prediction.
    X_ = dataframe
    if 'TARGET' in dataframe.columns:
        X_ = dataframe.loc[:, dataframe.columns != 'TARGET']

    return [dataframe, X_]
