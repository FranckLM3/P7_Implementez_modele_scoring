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

    def transfrom(self, client_id:dict):
        '''Preprocess the features for prediction
        '''
        try: 
            # Read data
            df = pd.read_csv('P7_Implementez_modele_scoring/data/application_train_sample.csv',
                            engine='pyarrow',
                            verbose=False,
                            encoding='ISO-8859-1',
                            )
            id = client_id['id']
            df = df[df['SK_ID_CURR'] == id]

            # NaN values for DAYS_EMPLOYED: 365.243 -> nan
            df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
            # Some simple new features (percentages)
            df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
            df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
            df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
            df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
            df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

            # Categorical features with Binary encode (0 or 1; two categories)
            for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
                df[bin_feature], uniques = pd.factorize(df[bin_feature])

            # Remove _MODE and _MEDI and FLAG_DOCUMENT features (EDA)
            rm = []
            num_col = df.select_dtypes(include=np.number).columns.to_list()
            for col in df[num_col].columns:
                if re.search('_MODE|_MEDI|FLAG_DOCUMENT_', col):
                    rm.append(col)
            # Keep Total AREA MODE as it is not repeated
            rm.remove('TOTALAREA_MODE')

            # Remove unique ID
            rm.append('SK_ID_CURR')

            df.drop(rm, axis=1, inplace=True)

            X = df.drop('TARGET', axis=1)

            X = self.preprocessor.transform(X)
        except: 
            X = 'This client is not in the database...'

        return X

    def make_prediction(self, features)->str:
        '''Predicts the credit score.
        Argument:
            features: list
        
        return:
            cluster: str
        '''
        if isinstance(features, str):
            score = 'This client is not in the database...'
        else: 
            pred = self.clf.predict_proba(features)[:, 1]

            pred = (pred >= 0.08)[0]

            score = self.scorer_meaning[pred]

        return score