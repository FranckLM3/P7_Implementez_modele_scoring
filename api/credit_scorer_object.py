import numpy as np
import pandas as pd
import pickle
import re


class credit_scorer:
    '''Create a object to implement credit scoring.
    '''
    def __init__(self, preprocess_path:str, model_path:str):
        self.preprocessor = self.get_preprocess(preprocess_path)
        self.clf = self.get_model(model_path)
        self.scorer_meaning = {
            False : 'No payement difficulties',
            True : 'Payement difficulties'}
    
    def get_model(self, model_path:str):
        '''Open the pkl file which store the model.
        Arguments: 
            model_path: Path model with pkl extension
        
        Returns:
            model: Model object
        '''

        with open(model_path,"rb") as f:
            clf = pickle.load(f)
        
        return clf
    
    def get_preprocess(self, preprocess_path:str):
        '''Open the pkl file which store the scaler.
        Arguments: 
            scaler_path: Path scaler with pkl extension
        
        Returns:
            scaler: scaler object
        '''

        with open(preprocess_path,"rb") as f:
            preprocessor = pickle.load(f)
        
        return preprocessor

    def transfrom(self, data, client_id:dict):
        '''Preprocess the features for prediction
        '''

        # Read data
        df = data.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        id = client_id['id']
        df = df[df['SK_ID_CURR'] == id]

        X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = df['TARGET']

        X = self.preprocessor.transform(X)

        return X

    def make_prediction(self, features)->str:
        '''Predicts the credit score.
        Argument:
            features: list
        
        return:
            cluster: str
        '''

        prob = self.clf.predict_proba(features)[:, 1]

        pred = (prob >= 0.47)[0]

        score = self.scorer_meaning[pred]

        return prob, score