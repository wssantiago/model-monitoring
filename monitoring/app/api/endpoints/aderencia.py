"""Endpoint para cálculo de aderência."""
from utils.preprocessing import data_preprocessing
from utils.reads import read_model, read_db, read_db
from sklearn.preprocessing import OneHotEncoder
import sys
from fastapi import APIRouter
import pandas as pd
from scipy.stats import ks_2samp
sys.path.append('...')

# Setting the logger formatting and poiting to correct file.
import logging
logging.basicConfig(filename='../monitoring.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

router = APIRouter(prefix="/aderencia")


# This function adds some handle_unkown policy to the OneHotEncoder so that
# unseen data may pass through the pipeline and be predicted as desired
def config_model(model):
    logging.info("Model is being fitted again for handling possibly unknown categories from features...")
    try:
        model.steps[0][1].transformers[1][1].steps.pop(1)
        model.steps[0][1].transformers[1][1].steps.append(
            ['encoder', OneHotEncoder(handle_unknown='ignore')])

        # Model must be fit again to apply the transformations.
        train_df, X_train = read_db('train')
        y_train = train_df.TARGET
        model.fit(X_train, y_train)

        logging.info("Model pipeline successfully updated!")

        return model
    except Exception as err:
        logging.error(
            "Exception raised while trying to reconfig the model: " + err)


# This function reads the model, the requested dataset and performs KS test
# together with the reference dataset.
def calc_aderencia(path: dict):
    logging.info("Calculating the adherence over the reference test base and the input base...")
    model = read_model()
    model = config_model(model)

    # Reading requested dataset as DataFrame
    path_req_dataset = path['req-dataset']
    req_df = pd.read_csv(
        path_req_dataset, compression='gzip', header=0, low_memory=False)
    req_df, X_req = data_preprocessing(req_df)

    # Reading reference dataset as DataFrame
    _, X_test = read_db('test')

    # Defining the score distribution for both datasets
    logging.info("Predicting scores distribution for both reference and input bases...")
    score_req = model.predict_proba(X_req)
    score_test = model.predict_proba(X_test)

    # The scores are passed as parameters to this scipy function.
    # It executes the KS test and returns the metrics relative
    # to either being samples from different distributions or
    # from the same one.
    distance = ks_2samp(score_req[:, 1], score_test[:, 1])
    logging.info("Successfully performed the Kolmogorov-Smirnov test over both score distributions using scipy...")
    
    return {'KStest-result': {'statistic': distance[0], 'p-value': distance[1]}}
