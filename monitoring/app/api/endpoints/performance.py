"""Endpoint para cálculo de Performance."""
from utils.reads import read_model, read_db
from fastapi import APIRouter
from typing import List
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import json
import numpy as np

import logging
logging.basicConfig(filename='../monitoring.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

router = APIRouter(prefix="/performance")

mes = {1: 'JAN', 2: 'FEV', 3: 'MAR', 4: 'ABR', 5: 'MAI', 6: 'JUN',
       7: 'JUL', 8: 'AGO', 9: 'SET', 10: 'OUT', 11: 'NOV', 12: 'DEZ'}


def enhance_model(model):
    logging.info(
        "The POST specified the enhanced model. Starting model enhancing...")
    try:
        model.steps.pop(1)
        model.steps.append(['encoder', DecisionTreeClassifier(max_depth=8, min_samples_leaf=0.15371419169712677,
                           min_samples_split=0.2572078354486276, class_weight={0: 1.0, 1: 0.28})])

        logging.info(
            "Model is being fitted to the training data now considering different class weights...")
        train_df, X_train = read_db('train')
        y_train = train_df.TARGET
        model.fit(X_train, y_train)

        logging.info("Model successfully enhanced!")
        return model
    except Exception as err:
        logging.error(
            "Exception raised while trying to enhance the model: " + err)


def get_roc(json_data: List[dict], model_version):
    logging.info("Calculating the roc score...")
    model = read_model()

    if model_version == 'enhanced':
        model = enhance_model(model)

    registros = pd.DataFrame(json_data)
    registros.fillna(value=np.nan, inplace=True)

    X_test = registros.loc[:, registros.columns != 'TARGET']
    y_test = registros.TARGET

    logging.info("Predicting the score for the requested json data...")
    score = model.predict_proba(X_test)[:, 1]
    yhat = model.predict(X_test)
    roc = roc_auc_score(y_test, score)
    logging.info("Successfully performed the roc score calculation over the " +
                 model_version + " model!")

    return roc


def get_volumetria(json_data: List[dict]):
    logging.info("Calculating the volumetry for the requested json data...")
    try:
        registros = pd.DataFrame(json_data)
        registros.REF_DATE = registros.REF_DATE.str[5:7].astype(int)
        registros.REF_DATE = registros.REF_DATE.apply(lambda x: mes[x])
        volumetria = registros.REF_DATE.value_counts()
        volumetria_json = volumetria.to_json(orient='index')
        volumetria_parsed = json.loads(volumetria_json)

        logging.info(
            "Successfully defined the monthly volumetry over the requested registers!")
        return volumetria_parsed
    except KeyError as kerr:
        logging.error(
            "Exception raised while trying to define volumetry: " + kerr)


def calc_performance(model_version: str, json_data: List[dict]):
    if model_version in ['default', 'enhanced']:
        roc = get_roc(json_data, model_version)
        volumetria = get_volumetria(json_data)

        return {'volumetria': volumetria, 'roc_score': roc}
    return {'volumetria': {}, 'roc_score': 0.0}
