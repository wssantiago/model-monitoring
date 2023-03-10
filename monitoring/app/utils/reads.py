import pickle
from utils.preprocessing import data_preprocessing
import pandas as pd


def read_model():
    try:
        model_file = open('../model.pkl', 'rb')
        model = pickle.load(model_file)

        return model
    except FileNotFoundError as fnf:
        print(fnf)


def read_db(db):
    try:
        df = pd.read_csv('../../datasets/credit_01/' + db + '.gz',
                         compression='gzip', header=0, low_memory=False)
        test_df, X_test = data_preprocessing(df)

        return [test_df, X_test]
    except FileNotFoundError as fnf:
        print(fnf)
