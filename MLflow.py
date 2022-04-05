import numpy as np
import pandas as pd
import gc
import time
import re 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
import mlflow
import mlflow.lightgbm
from urllib.parse import urlparse

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
     # calculate the g-mean for each threshold
    g_means = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(g_means)
    thresholds_value = thresholds[ix]

    pred = (pred >= thresholds[ix]).astype(bool)

    auc = metrics.roc_auc_score(actual, pred)
    recall = metrics.recall_score(actual, pred)
    precision = metrics.precision_score(actual, pred)
    f1 = metrics.f1_score(actual, pred)

    return auc, recall, precision, f1, thresholds_value

def application_train_test():
    # Read data
    df = pd.read_csv('data/application_train.csv',
                    low_memory=False,
                    verbose=False,
                    encoding='ISO-8859-1',
                    dtype={'Special': 'object'}
                    )

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
    y = df['TARGET']


    # Categorical features with One-Hot encode
    categorical_features = X.select_dtypes(exclude=np.number).columns
    steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
    categorical_transformer = Pipeline(steps=steps)

    # Numeric features with StandardScaler
    numeric_features = X.select_dtypes(include=np.number).columns
    steps = [('scaler', StandardScaler())]
    numeric_transformer = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                    ])

    X = preprocessor.fit_transform(X)

    # Save feature names to future features importance eval
    onehot_columns = list(preprocessor.named_transformers_['cat'].named_steps['encoder'].\
                          get_feature_names_out(input_features=categorical_features))
    numeric_features_list = list(numeric_features)
    feats = np.concatenate((numeric_features_list, onehot_columns))

    df_encoded = pd.DataFrame(X, columns=feats)
    df_encoded['TARGET'] = df['TARGET']

    # Rename columns to avoid JSON character
    charac = ','
    charac_2 = ':'
    new_list = []
    for item in df_encoded.columns: 
        if (charac in item) | (charac_2 in item):
            item = re.sub(charac, '_', item)
            item = re.sub(charac_2, '_', item)
            new_list.append(item)
        else:
            new_list.append(item)
    df_encoded = df_encoded.set_axis(new_list, axis=1)

    # Train, Test split
    df_train, df_test = train_test_split(df_encoded,
                                         test_size=.2,
                                         random_state=123)

    print("Train samples: {}, test samples: {}".format(df_train.shape, df_test.shape))
    del df
    gc.collect()
    return df_train, df_test

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        df_train, df_test = application_train_test()
    except Exception as e:
        logger.exception(
            "Unable to load training & test CSV. Error: %s", e
        )

    # Model fitting for MLflow
    train_x = df_train.drop([('TARGET')], axis=1)
    test_x = df_test.drop([('TARGET')], axis=1)
    train_y = df_train['TARGET']
    test_y = df_test[['TARGET']]

    n_estimators=1000
    learning_rate=0.02
    num_leaves=44
    colsample_bytree=0.2782455842129526
    subsample=0.9835209048870284
    max_depth=8
    reg_alpha=0.29323776895986786
    reg_lambda=00.8944170104864534
    min_split_gain=0.13787924996910783
    min_child_weight=44.68845496978565

    with mlflow.start_run():
        clf = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            silent=-1,
            verbose=-1,
            )

        clf.fit(train_x, train_y)

        pred_y = clf.predict_proba(test_x)[:, 1]

        auc, recall, precision, f1, thresholds_value = eval_metrics(test_y, pred_y)

        print("""lightGBM model :
            n estimators=%d
            learning rate=%f
            num leaves=%d
            colsample bytree=%f
            subsample=%f
            max depth=%d
            reg alpha=%f
            reg lambda=%f
            min split gain=%f
            min child weight=%f""" % (n_estimators,
                                    learning_rate,
                                    num_leaves,
                                    colsample_bytree,
                                    subsample,
                                    max_depth,
                                    reg_alpha,
                                    reg_lambda,
                                    min_split_gain,
                                    min_child_weight))
        
        print("  AUC: %s" % auc)
        print("  Recall: %s" % recall)
        print("  Precision: %s" % precision)
        print("  f1: %s" % f1)
        print("  Threshold: %s" % thresholds_value)

        mlflow.log_param("n estimators", n_estimators)
        mlflow.log_param("learning rate", learning_rate)
        mlflow.log_param("num leaves", num_leaves)
        mlflow.log_param("colsample bytree", colsample_bytree)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("max depth", max_depth)
        mlflow.log_param("reg alpha", reg_alpha)
        mlflow.log_param("reg lambda", reg_lambda)
        mlflow.log_param("min split gain", min_split_gain)
        mlflow.log_param("rmin child weight", min_child_weight)

        mlflow.log_metric("Thresold", thresholds_value)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("f1", f1)
        

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(clf, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(clf, "model")