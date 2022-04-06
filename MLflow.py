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
from sklearn.feature_selection import SelectFromModel

import warnings
import mlflow
import mlflow.lightgbm
from urllib.parse import urlparse

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Read data
df = pd.read_csv('data/model_dataset.csv',
            nrows=10000,
            low_memory=False,
            verbose=False,
            encoding='ISO-8859-1',
            dtype={'Special': 'object'}
            )

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

def preprocessing(df):
    # Read data
    
    df = df.replace([np.inf, -np.inf], np.nan)

    X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
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

    # Feature selection
    sfm = SelectFromModel(LGBMClassifier(n_jobs=-1))
   
    sfm.fit(X, y)
    X = sfm.transform(X)

    print('X shape after features selection: ', X.shape)
    
    # Save feature names to future features importance eval
    onehot_columns = list(preprocessor.named_transformers_['cat'].named_steps['encoder'].\
                          get_feature_names_out(input_features=categorical_features))
    numeric_features_list = list(numeric_features)
    feats = np.concatenate((numeric_features_list, onehot_columns))
    feats = feats[sfm.get_support()]

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
    #warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        df_train, df_test = preprocessing(df)
    except Exception as e:
        logger.exception(
            "Unable to load training & test CSV. Error: %s", e
        )

    # Model fitting for MLflow
    train_x = df_train.drop(['TARGET'], axis=1)
    test_x = df_test.drop(['TARGET'], axis=1)
    train_y = df_train['TARGET']
    test_y = df_test[['TARGET']]

    n_estimators=1000
    learning_rate=0.02
    colsample_bytree=0.2352167642701535
    max_depth=9
    min_child_weight=34.67802513470199
    min_split_gain=0.772886577362599
    num_leaves=44
    reg_alpha=0.6466454272437308
    reg_lambda=0.883716377059081
    subsample=0.1498951237319723
    scale_pos_weight=11.387150050352467

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
            scale_pos_weight=scale_pos_weight,
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
        mlflow.log_param("min child weight", min_child_weight)
        mlflow.log_param("Scale pos weight", scale_pos_weight)

        mlflow.log_metric("Thresold", thresholds_value)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("f1", f1)
        

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            mlflow.sklearn.log_model(clf, "classifier", registered_model_name="LightGBM classifier")
        else:
            mlflow.sklearn.log_model(clf, "classifier")