#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:35:50 2023

@author: rayhaan
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# from pycaret.classification import *
# from pycaret.clustering import *
import time
import itertools
import random

from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.deep_svdd import DeepSVDD

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from pyod.models.xgbod import XGBOD
from deepod.models.feawad import FeaWAD
from deepod.models.dsad import DeepSAD
from deepod.models.prenet import PReNet
from deepod.models.devnet import DevNet

bondora_df = pd.read_csv('Bondora_preprocessed.csv')

bondora_df.head()

# Dropping irrelevant columns
irrelevant_cols = ['LanguageCode', 'Country', 'County', 'City']
bondora_df = bondora_df.drop(columns=irrelevant_cols)

# Transforming NewCreditCustomer column to boolean
bondora_df['NewCreditCustomer'] = bondora_df['NewCreditCustomer'].apply(
    lambda x: 1 if x == 'New_credit_Customer' else 0)

# Transforming Restructured column to boolean
bondora_df['Restructured'] = bondora_df['Restructured'].apply(
    lambda x: 1 if x == 'Yes' else 0)

# Converting date columns to age columns (in months)
date_cols = ['LoanDate', 'FirstPaymentDate', 'LastPaymentOn']
maturity_date_cols = ['MaturityDate_Original', 'MaturityDate_Last']

def date_to_months_since(x):
    if str(x) == 'nan':
        return None
    return round((datetime.today() - 
                  datetime.strptime(x, '%Y-%m-%d')).days/30)

for date_col in date_cols:
    bondora_df['MonthsSince' + date_col] = \
        bondora_df[date_col].apply(date_to_months_since)
    
for date_col in maturity_date_cols:
    bondora_df['MonthsUntil' + date_col] = \
        bondora_df[date_col].apply(date_to_months_since)
        
bondora_df = bondora_df.drop(columns=date_cols + maturity_date_cols)

# Encoding ordinal categories
ordinal_category_cols = ['VerificationType', 'Education', 'EmploymentStatus',
                         'EmploymentDurationCurrentEmployer',
                         'HomeOwnershipType', 'Rating', 'CreditScoreEsMicroL']
ordinal_category_scores = {
    'VerificationType': {
        'Income_unverified_crossref_phone': 0,
        'Income_unverified': 0,
        'Income_expenses_verified': 1,
        'Income_verified': 1,
        'Not_set': None
        },
    'Education': {
        'Not_present': 0,
        'Primary': 1,
        'Basic': 2,
        'Vocational': 3,
        'Secondary': 4,
        'Higher': 5
        },
    'EmploymentStatus': {
        'Not_specified': 0,
        'Unemployed': 1,
        'Partially': 2,
        'Fully': 3,
        'Self_employed': 4,
        'Entrepreneur': 5,
        'Retiree': 6
        },
    'EmploymentDurationCurrentEmployer': {
        'TrialPeriod': 0,
        'UpTo1Year': 1,
        'UpTo2Years': 2,
        'UpTo3Years': 3,
        'UpTo4Years': 4,
        'UpTo5Years': 5,
        'MoreThan5Years': 6,
        'Retiree': 7,
        'Other': 8
        },
    'HomeOwnershipType': {
        'Not_specified': None,
        'Homeless': 0,
        'Tenant_unfurnished_property': 1,
        'Tenant_pre_furnished_property': 1,
        'Joint_tenant': 1,
        'Living_with_parents': 1,
        'Council_house': 1,
        'Mortgage': 2,
        'Joint_ownership': 2,
        'Owner': 2,
        'Owner_with_encumbrance': 2,
        'Other': 3
        },
    'Rating': {
        'HR': 0,
        'F': 1,
        'E': 2,
        'D': 3,
        'C': 4,
        'B': 5,
        'A': 6,
        'AA': 7
        },
    'CreditScoreEsMicroL': {
        'M': None,
        'M10': 0,
        'M9': 1,
        'M8': 2,
        'M7': 3,
        'M6': 4,
        'M5': 5,
        'M4': 6,
        'M3': 7,
        'M2': 8,
        'M1': 9
        }
    }

for category in ordinal_category_cols:
    category_score = ordinal_category_scores[category]
    bondora_df[category] = bondora_df[category].map(category_score)

# Encoding other categorical attributes
nominal_category_cols = ['Gender', 'UseOfLoan', 'MaritalStatus', 'OccupationArea']

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

for feature in nominal_category_cols:
    bondora_df = encode_and_bind(bondora_df, feature)

bondora_df = bondora_df.drop(columns=['CreditScoreEsMicroL']) # Too many NaNs
bondora_df['PreviousRepaymentsBeforeLoan'] = \
    bondora_df['PreviousRepaymentsBeforeLoan'].fillna(0)
bondora_df['MonthsSinceLastPaymentOn'] = \
    bondora_df['MonthsSinceLastPaymentOn'].fillna(bondora_df['MonthsSinceFirstPaymentDate'])[15]
bondora_df = bondora_df.dropna()
bondora_df.reset_index(inplace=True, drop=True)


X = bondora_df.drop(columns=['Default'])
y = bondora_df['Default']

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train = X_train
# train['Default'] = y_train

# test = X_test
# test['Default'] = y_test

# exp_setup = setup(train, normalize = True, target = 'Default')
# # exp_setup = setup(train, normalize = True) #, target = 'Default')

# top5 = compare_models(n_select = 10)

# for model in top5:
#     print(model)
    
#     start_time = time.time()
    
#     predictions = predict_model(model, data=test.drop('Default', axis=1))

#     print(accuracy_score(test["Default"], predictions["Label"]))
#     print(precision_recall_fscore_support(test["Default"], predictions["Label"], average=None, labels=[0,1]))
#     # print(accuracy_icing_event(test["Default"], predictions["Label"]))
    
#     end_time = time.time()
    
#     print("Prediction time: %s sec" % (end_time - start_time))

############################ CLUSTERING ############################

# exp_setup = setup(train.drop(columns=['Default']), normalize = True) #, target = 'Default')

# models()

# dbscan = create_model('dbscan', num_clusters = 2)

# start_time = time.time()

# results = assign_model(dbscan)
# results.head(10)

# end_time = time.time()

# print("Prediction time: %s sec" % (end_time - start_time))

# dbscan = DBSCAN(eps=0.5, min_samples=5)
# dbscan.fit(train.drop(columns=['Default']))

# start_time = time.time()
# x = dbscan.labels_
# # y_pred = [0 if i <= 0 else 1 for i in x]
# y_pred = [0 if i < np.mean(x) else 1 for i in x]

# end_time = time.time()

# print("Prediction time: %s sec" % (end_time - start_time))

# print(accuracy_score(train["Default"], y_pred))
# print(precision_recall_fscore_support(train["Default"], y_pred, average=None, labels=[0,1]))

############################ CLUSTERING ############################

lr_model = LogisticRegression(solver='liblinear', random_state=123)
lr_model.fit(X_scaled, y_train)

print("Performing randomized Cross Validation on the Logistic Regression model...")

lr_param_grid = {
    'penalty' : ['l1', 'l2'],
    'C' : [0.001,0.01,0.1,1,10,100]
}

lr_cv_model = RandomizedSearchCV(lr_model, param_distributions=lr_param_grid, cv=5)
lr_cv_model.fit(X_scaled, y_train)

print("The following are the optimal parameters after cross validation:")
print(lr_cv_model.best_params_)

penalty = lr_cv_model.best_params_['penalty']
C = lr_cv_model.best_params_['C']

lr_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

lr_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

lr_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running Logistic Regression model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    lr_model = LogisticRegression(solver='liblinear', penalty=penalty, C=C, 
                                  random_state=123)
    lr_model.fit(X_scaled, y_train)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = lr_model.predict(X_test_scaled)
    # y_pred = (lr_model.predict_proba(X_test)[:,1] >= sum(y_test) / len(y_test)).astype(int)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    lr_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    lr_metrics['precision'].append(precision_score(y_test, y_pred))
    lr_metrics['recall'].append(recall_score(y_test, y_pred))
    lr_metrics['f1'].append(f1_score(y_test, y_pred))
    lr_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    lr_metrics['training_time'].append(training_time)
    lr_metrics['pred_time'].append(pred_time)
    
    lr_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    lr_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    lr_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    lr_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    lr_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    lr_metrics_30['training_time'].append(training_time)
    lr_metrics_30['pred_time'].append(pred_time)
    
    lr_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    lr_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    lr_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    lr_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    lr_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    lr_metrics_70['training_time'].append(training_time)
    lr_metrics_70['pred_time'].append(pred_time)

lr_metrics_mean = {k:np.mean(v) for k, v in lr_metrics.items()}
print("Logistic Regression model mean metrics:")
print(lr_metrics_mean)

lr_metrics_30_mean = {k:np.mean(v) for k, v in lr_metrics_30.items()}
print("Logistic Regression model mean metrics with 30% threshold:")
print(lr_metrics_30_mean)

lr_metrics_70_mean = {k:np.mean(v) for k, v in lr_metrics_70.items()}
print("Logistic Regression model mean metrics with 70% threshold:")
print(lr_metrics_70_mean)

sgd_model = SGDClassifier(random_state=123)
sgd_model.fit(X_scaled, y_train)

sgd_params = {
    'loss': ['log_loss', 'modified_huber'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    }

print("Performing randomized Cross Validation on the SGD model...")

sgd_cv_model = RandomizedSearchCV(sgd_model, param_distributions=sgd_params, cv=5)
sgd_cv_model.fit(X_scaled, y_train)

print("The following are the optimal parameters after cross validation:")
print(sgd_cv_model.best_params_)

loss = sgd_cv_model.best_params_['loss']
penalty = sgd_cv_model.best_params_['penalty']
alpha = sgd_cv_model.best_params_['alpha']
learning_rate = sgd_cv_model.best_params_['learning_rate']

sgd_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

sgd_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

sgd_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running SGD model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    sgd_model = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha,
                              learning_rate=learning_rate, random_state=123)
    sgd_model.fit(X_scaled, y_train)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = sgd_model.predict(X_test_scaled)
    y_pred_proba = sgd_model.predict_proba(X_test_scaled)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    sgd_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    sgd_metrics['precision'].append(precision_score(y_test, y_pred))
    sgd_metrics['recall'].append(recall_score(y_test, y_pred))
    sgd_metrics['f1'].append(f1_score(y_test, y_pred))
    sgd_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    sgd_metrics['training_time'].append(training_time)
    sgd_metrics['pred_time'].append(pred_time)
    
    sgd_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    sgd_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    sgd_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    sgd_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    sgd_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    sgd_metrics_30['training_time'].append(training_time)
    sgd_metrics_30['pred_time'].append(pred_time)
    
    sgd_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    sgd_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    sgd_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    sgd_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    sgd_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    sgd_metrics_70['training_time'].append(training_time)
    sgd_metrics_70['pred_time'].append(pred_time)

sgd_metrics_mean = {k:np.mean(v) for k, v in sgd_metrics.items()}
print("SGD model mean metrics:")
print(sgd_metrics_mean)

sgd_metrics_30_mean = {k:np.mean(v) for k, v in sgd_metrics_30.items()}
print("SGD model mean metrics with 30% threshold:")
print(sgd_metrics_30_mean)

sgd_metrics_70_mean = {k:np.mean(v) for k, v in sgd_metrics_70.items()}
print("SGD model mean metrics with 70% threshold:")
print(sgd_metrics_70_mean)

# clf = IForest(n_estimators = 500, contamination = 0.23, random_state=123)
# clf.fit(X_train)

xgb_model = XGBClassifier(random_state=123)
xgb_model.fit(X_scaled, y_train)

xgb_params = {
    'n_estimators': [100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [1, 2, 4, 5],
    'min_samples_leaf': [1, 2, 4, 5],
    'max_leaf_nodes': [4, 10, 20, 50, None]
    }

print("Performing randomized Cross Validation on the XGB model...")

xgb_cv_model = RandomizedSearchCV(xgb_model, param_distributions=xgb_params, cv=5)
xgb_cv_model.fit(X_scaled, y_train)

print("The following are the optimal parameters after cross validation:")
print(xgb_cv_model.best_params_)

n_estimators = xgb_cv_model.best_params_['n_estimators']
criterion = xgb_cv_model.best_params_['criterion']
min_samples_split = xgb_cv_model.best_params_['min_samples_split']
min_samples_leaf = xgb_cv_model.best_params_['min_samples_leaf']
max_leaf_nodes = xgb_cv_model.best_params_['max_leaf_nodes']

xgb_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

xgb_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

xgb_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running XGB model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()

    n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes
    
    xgb_model = XGBClassifier(n_estimators=n_estimators, criterion=criterion,
                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                              max_leaf_nodes=max_leaf_nodes, random_state=123)
    xgb_model.fit(X_scaled, y_train)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    xgb_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    xgb_metrics['precision'].append(precision_score(y_test, y_pred))
    xgb_metrics['recall'].append(recall_score(y_test, y_pred))
    xgb_metrics['f1'].append(f1_score(y_test, y_pred))
    xgb_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    xgb_metrics['training_time'].append(training_time)
    xgb_metrics['pred_time'].append(pred_time)
    
    xgb_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    xgb_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    xgb_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    xgb_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    xgb_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    xgb_metrics_30['training_time'].append(training_time)
    xgb_metrics_30['pred_time'].append(pred_time)
    
    xgb_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    xgb_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    xgb_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    xgb_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    xgb_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    xgb_metrics_70['training_time'].append(training_time)
    xgb_metrics_70['pred_time'].append(pred_time)

xgb_metrics_mean = {k:np.mean(v) for k, v in xgb_metrics.items()}
print("XGB model mean metrics:")
print(xgb_metrics_mean)

xgb_metrics_30_mean = {k:np.mean(v) for k, v in xgb_metrics_30.items()}
print("XGB model mean metrics with 30% threshold:")
print(xgb_metrics_30_mean)

xgb_metrics_70_mean = {k:np.mean(v) for k, v in xgb_metrics_70.items()}
print("XGB model mean metrics with 70% threshold:")
print(xgb_metrics_70_mean)

lgbm_model = LGBMClassifier(random_state=123)
lgbm_model.fit(X_scaled, y_train)

lgbm_params = {
    'num_leaves': [5, 10, 20, 50],
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample': [0.7, 0.8, 0.9],
    'min_child_samples': [10, 50, 100],
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
    }

print("Performing randomized Cross Validation on the LGBM model...")

lgbm_cv_model = RandomizedSearchCV(lgbm_model, param_distributions=lgbm_params, cv=5, random_state=123)
lgbm_cv_model.fit(X_scaled, y_train)

print("The following are the optimal parameters after cross validation:")
print(lgbm_cv_model.best_params_)

num_leaves = lgbm_cv_model.best_params_['num_leaves']
n_estimators = lgbm_cv_model.best_params_['n_estimators']
max_depth = lgbm_cv_model.best_params_['max_depth']
colsample_bytree = lgbm_cv_model.best_params_['colsample_bytree']
subsample = lgbm_cv_model.best_params_['subsample']
min_child_samples = lgbm_cv_model.best_params_['min_child_samples']
min_child_weight = lgbm_cv_model.best_params_['min_child_weight']
reg_alpha = lgbm_cv_model.best_params_['reg_alpha']
reg_lambda = lgbm_cv_model.best_params_['reg_lambda']

lgbm_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

lgbm_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

lgbm_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running LGBM model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()

    lgbm_model = LGBMClassifier(num_leaves=num_leaves, n_estimators=n_estimators, max_depth=max_depth,\
                                colsample_bytree=colsample_bytree, subsample=subsample, min_child_samples=min_child_samples,
                                min_child_weight=min_child_weight, reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=123)
    lgbm_model.fit(X_scaled, y_train)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = lgbm_model.predict(X_test_scaled)
    y_pred_proba = lgbm_model.predict_proba(X_test_scaled)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    lgbm_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    lgbm_metrics['precision'].append(precision_score(y_test, y_pred))
    lgbm_metrics['recall'].append(recall_score(y_test, y_pred))
    lgbm_metrics['f1'].append(f1_score(y_test, y_pred))
    lgbm_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    lgbm_metrics['training_time'].append(training_time)
    lgbm_metrics['pred_time'].append(pred_time)
    
    lgbm_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    lgbm_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    lgbm_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    lgbm_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    lgbm_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    lgbm_metrics_30['training_time'].append(training_time)
    lgbm_metrics_30['pred_time'].append(pred_time)
    
    lgbm_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    lgbm_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    lgbm_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    lgbm_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    lgbm_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    lgbm_metrics_70['training_time'].append(training_time)
    lgbm_metrics_70['pred_time'].append(pred_time)

lgbm_metrics_mean = {k:np.mean(v) for k, v in lgbm_metrics.items()}
print("LGBM model mean metrics:")
print(lgbm_metrics_mean)

lgbm_metrics_30_mean = {k:np.mean(v) for k, v in lgbm_metrics_30.items()}
print("LGBM model mean metrics with 30% threshold:")
print(lgbm_metrics_30_mean)

lgbm_metrics_70_mean = {k:np.mean(v) for k, v in lgbm_metrics_70.items()}
print("LGBM model mean metrics with 70% threshold:")
print(lgbm_metrics_70_mean)

cb_model = CatBoostClassifier(random_state=123)
cb_model.fit(X_scaled, y_train)

cb_params = {
    'depth':[1, 2, 5, 10],
    'iterations': [100, 200, 500, 1000],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3], 
    'l2_leaf_reg': [1, 5, 10, 50],
    'border_count': [5, 10, 20, 50, 100, 200]
    }

print("Performing randomized Cross Validation on the CatBoost model...")

cb_cv_model = RandomizedSearchCV(cb_model, param_distributions=cb_params, cv=5, random_state=123)
cb_cv_model.fit(X_scaled, y_train)

print("The following are the optimal parameters after cross validation:")
print(cb_cv_model.best_params_)

depth = cb_cv_model.best_params_['depth']
iterations = cb_cv_model.best_params_['iterations']
learning_rate = cb_cv_model.best_params_['learning_rate']
l2_leaf_reg = cb_cv_model.best_params_['l2_leaf_reg']
border_count = cb_cv_model.best_params_['border_count']

cb_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

cb_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

cb_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running CatBoost model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    cb_model = CatBoostClassifier(depth=depth, iterations=iterations, learning_rate=learning_rate,
                                  l2_leaf_reg=l2_leaf_reg, border_count=border_count, random_state=123)
    cb_model.fit(X_scaled, y_train)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = cb_model.predict(X_test_scaled)
    y_pred_proba = cb_model.predict_proba(X_test_scaled)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    cb_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    cb_metrics['precision'].append(precision_score(y_test, y_pred))
    cb_metrics['recall'].append(recall_score(y_test, y_pred))
    cb_metrics['f1'].append(f1_score(y_test, y_pred))
    cb_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    cb_metrics['training_time'].append(training_time)
    cb_metrics['pred_time'].append(pred_time)
    
    cb_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    cb_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    cb_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    cb_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    cb_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    cb_metrics_30['training_time'].append(training_time)
    cb_metrics_30['pred_time'].append(pred_time)
    
    cb_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    cb_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    cb_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    cb_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    cb_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    cb_metrics_70['training_time'].append(training_time)
    cb_metrics_70['pred_time'].append(pred_time)

cb_metrics_mean = {k:np.mean(v) for k, v in cb_metrics.items()}
print("CatBoost model mean metrics:")
print(cb_metrics_mean)

cb_metrics_30_mean = {k:np.mean(v) for k, v in cb_metrics_30.items()}
print("CatBoost model mean metrics with 30% threshold:")
print(cb_metrics_30_mean)

cb_metrics_70_mean = {k:np.mean(v) for k, v in cb_metrics_70.items()}
print("CatBoost model mean metrics with 70% threshold:")
print(cb_metrics_70_mean)

if_model = IForest(contamination = 0.5, random_state=123)
if_model.fit(X_scaled)

if_params = {
    'n_estimators': [100, 200, 500, 1000]
    }

print("Performing randomized Cross Validation on the IForest model...")

if_cv_model = RandomizedSearchCV(if_model, param_distributions=if_params, cv=5,
                                 scoring='roc_auc', random_state=123)
if_cv_model.fit(X_scaled, y_train)

print("The following are the optimal parameters after cross validation:")
print(if_cv_model.best_params_)

n_estimators = if_cv_model.best_params_['n_estimators']

if_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

if_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

if_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running IForest model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    if_model = IForest(n_estimators=n_estimators, contamination = 0.5,
                       random_state=123)
    if_model.fit(X_scaled)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = if_model.predict(X_test_scaled)
    y_pred_proba = if_model.predict_proba(X_test_scaled)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    if_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    if_metrics['precision'].append(precision_score(y_test, y_pred))
    if_metrics['recall'].append(recall_score(y_test, y_pred))
    if_metrics['f1'].append(f1_score(y_test, y_pred))
    if_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    if_metrics['training_time'].append(training_time)
    if_metrics['pred_time'].append(pred_time)
    
    if_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    if_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    if_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    if_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    if_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    if_metrics_30['training_time'].append(training_time)
    if_metrics_30['pred_time'].append(pred_time)
    
    if_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    if_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    if_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    if_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    if_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    if_metrics_70['training_time'].append(training_time)
    if_metrics_70['pred_time'].append(pred_time)

if_metrics_mean = {k:np.mean(v) for k, v in if_metrics.items()}
print("IForest model mean metrics:")
print(if_metrics_mean)

if_metrics_30_mean = {k:np.mean(v) for k, v in if_metrics_30.items()}
print("IForest model mean metrics with 30% threshold:")
print(if_metrics_30_mean)

if_metrics_70_mean = {k:np.mean(v) for k, v in if_metrics_70.items()}
print("IForest model mean metrics with 70% threshold:")
print(if_metrics_70_mean)

abod_model = ABOD(contamination = 0.5)
abod_model.fit(X_scaled)

abod_params = {
    'n_neighbors': [1, 2, 5, 10]
    }

print("Performing randomized Cross Validation on the ABOD model...")

abod_cv_model = RandomizedSearchCV(abod_model, param_distributions=abod_params, cv=5,
                                 scoring='roc_auc', random_state=123)
abod_cv_model.fit(X_scaled, y_train)

print("The following are the optimal parameters after cross validation:")
print(abod_cv_model.best_params_)

n_neighbors = abod_cv_model.best_params_['n_neighbors']

abod_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

abod_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

abod_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running ABOD model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    abod_model = ABOD(n_neighbors=n_neighbors, contamination = 0.5)
    abod_model.fit(X_scaled)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = abod_model.predict(X_test_scaled)
    y_pred_proba = abod_model.predict_proba(X_test)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    abod_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    abod_metrics['precision'].append(precision_score(y_test, y_pred))
    abod_metrics['recall'].append(recall_score(y_test, y_pred))
    abod_metrics['f1'].append(f1_score(y_test, y_pred))
    abod_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    abod_metrics['training_time'].append(training_time)
    abod_metrics['pred_time'].append(pred_time)
    
    abod_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    abod_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    abod_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    abod_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    abod_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    abod_metrics_30['training_time'].append(training_time)
    abod_metrics_30['pred_time'].append(pred_time)
    
    abod_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    abod_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    abod_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    abod_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    abod_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    abod_metrics_70['training_time'].append(training_time)
    abod_metrics_70['pred_time'].append(pred_time)
    
abod_metrics_mean = {k:np.mean(v) for k, v in abod_metrics.items()}
print("ABOD model mean metrics:")
print(abod_metrics_mean)

abod_metrics_30_mean = {k:np.mean(v) for k, v in abod_metrics_30.items()}
print("ABOD model mean metrics with 30% threshold:")
print(abod_metrics_30_mean)

abod_metrics_70_mean = {k:np.mean(v) for k, v in abod_metrics_70.items()}
print("ABOD model mean metrics with 70% threshold:")
print(abod_metrics_70_mean)

ecod_model = ECOD(contamination = 0.5)
ecod_model.fit(X_scaled)

ecod_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

ecod_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

ecod_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running ECOD model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    ecod_model = ECOD(contamination = 0.5)
    ecod_model.fit(X_scaled)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = ecod_model.predict(X_test_scaled)
    y_pred_proba = ecod_model.predict_proba(X_test)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    ecod_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    ecod_metrics['precision'].append(precision_score(y_test, y_pred))
    ecod_metrics['recall'].append(recall_score(y_test, y_pred))
    ecod_metrics['f1'].append(f1_score(y_test, y_pred))
    ecod_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    ecod_metrics['training_time'].append(training_time)
    ecod_metrics['pred_time'].append(pred_time)
    
    ecod_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    ecod_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    ecod_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    ecod_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    ecod_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    ecod_metrics_30['training_time'].append(training_time)
    ecod_metrics_30['pred_time'].append(pred_time)
    
    ecod_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    ecod_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    ecod_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    ecod_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    ecod_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    ecod_metrics_70['training_time'].append(training_time)
    ecod_metrics_70['pred_time'].append(pred_time)
    
ecod_metrics_mean = {k:np.mean(v) for k, v in ecod_metrics.items()}
print("ECOD model mean metrics:")
print(ecod_metrics_mean)

ecod_metrics_30_mean = {k:np.mean(v) for k, v in ecod_metrics_30.items()}
print("ECOD model mean metrics with 30% threshold:")
print(ecod_metrics_30_mean)

ecod_metrics_70_mean = {k:np.mean(v) for k, v in ecod_metrics_70.items()}
print("ECOD model mean metrics with 70% threshold:")
print(ecod_metrics_70_mean)

copod_model = COPOD(contamination = 0.5)
copod_model.fit(X_scaled)

copod_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

copod_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

copod_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running COPOD model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    copod_model = COPOD(contamination = 0.5)
    copod_model.fit(X_scaled)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = copod_model.predict(X_test_scaled)
    y_pred_proba = copod_model.predict_proba(X_test)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    copod_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    copod_metrics['precision'].append(precision_score(y_test, y_pred))
    copod_metrics['recall'].append(recall_score(y_test, y_pred))
    copod_metrics['f1'].append(f1_score(y_test, y_pred))
    copod_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    copod_metrics['training_time'].append(training_time)
    copod_metrics['pred_time'].append(pred_time)
    
    copod_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    copod_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    copod_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    copod_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    copod_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    copod_metrics_30['training_time'].append(training_time)
    copod_metrics_30['pred_time'].append(pred_time)
    
    copod_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    copod_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    copod_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    copod_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    copod_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    copod_metrics_70['training_time'].append(training_time)
    copod_metrics_70['pred_time'].append(pred_time)
    
copod_metrics_mean = {k:np.mean(v) for k, v in copod_metrics.items()}
print("COPOD model mean metrics:")
print(copod_metrics_mean)

copod_metrics_30_mean = {k:np.mean(v) for k, v in copod_metrics_30.items()}
print("COPOD model mean metrics with 30% threshold:")
print(copod_metrics_30_mean)

copod_metrics_70_mean = {k:np.mean(v) for k, v in copod_metrics_70.items()}
print("COPOD model mean metrics with 70% threshold:")
print(copod_metrics_70_mean)

deepsvdd_model = DeepSVDD()
deepsvdd_model.fit(X_scaled)

# deepsvdd_params = {
#     'use_ae': [True, False],
#     'hidden_activation': ['sigmoid', 'relu'],
#     'output_activation': ['sigmoid', 'relu'],
#     'optimizer': ['adam', 'rmsprop', 'sgd'],
#     'epochs': [10, 20, 50, 100],
#     'batch_size': [16, 32, 64, 128],
#     'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
#     }

# deepsvdd_params = {
#     'use_ae': [True, False],
#     'epochs': [10, 50],
#     'batch_size': [16, 32],
#     'dropout_rate': [0.1, 0.2]
#     }

# print("Performing randomized Cross Validation on the DeepSVDD model...")

# deepsvdd_cv_model = RandomizedSearchCV(deepsvdd_model, param_distributions=deepsvdd_params, cv=5,
#                                  scoring='roc_auc')
# deepsvdd_cv_model.fit(X_scaled)

# print("The following are the optimal parameters after cross validation:")
# print(deepsvdd_cv_model.best_params_)

# n_neighbors = deepsvdd_cv_model.best_params_['n_neighbors']

deepsvdd_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

deepsvdd_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

deepsvdd_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running DeepSVDD model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    deepsvdd_model = DeepSVDD()
    deepsvdd_model.fit(X_scaled)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = deepsvdd_model.predict(X_test_scaled)
    y_pred_proba = deepsvdd_model.predict_proba(X_test)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    deepsvdd_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    deepsvdd_metrics['precision'].append(precision_score(y_test, y_pred))
    deepsvdd_metrics['recall'].append(recall_score(y_test, y_pred))
    deepsvdd_metrics['f1'].append(f1_score(y_test, y_pred))
    deepsvdd_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    deepsvdd_metrics['training_time'].append(training_time)
    deepsvdd_metrics['pred_time'].append(pred_time)
    
    deepsvdd_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    deepsvdd_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    deepsvdd_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    deepsvdd_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    deepsvdd_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    deepsvdd_metrics_30['training_time'].append(training_time)
    deepsvdd_metrics_30['pred_time'].append(pred_time)
    
    deepsvdd_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    deepsvdd_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    deepsvdd_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    deepsvdd_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    deepsvdd_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    deepsvdd_metrics_70['training_time'].append(training_time)
    deepsvdd_metrics_70['pred_time'].append(pred_time)
    
deepsvdd_metrics_mean = {k:np.mean(v) for k, v in deepsvdd_metrics.items()}
print("DeepSVDD model mean metrics:")
print(deepsvdd_metrics_mean)

deepsvdd_metrics_30_mean = {k:np.mean(v) for k, v in deepsvdd_metrics_30.items()}
print("DeepSVDD model mean metrics with 30% threshold:")
print(deepsvdd_metrics_30_mean)

deepsvdd_metrics_70_mean = {k:np.mean(v) for k, v in deepsvdd_metrics_70.items()}
print("DeepSVDD model mean metrics with 70% threshold:")
print(deepsvdd_metrics_70_mean)

# Semi-supervised

xgbod_model = XGBOD(random_state=123)
xgbod_model.fit(X_scaled, y_train)

xgbod_params = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3], 
    'max_depth':[1, 2, 5, 10],
    'n_estimators': [50, 100, 150],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
    }

print("Performing randomized Cross Validation on the XGBOD model...")

xgbod_cv_model = RandomizedSearchCV(xgbod_model, param_distributions=xgbod_params, cv=5, random_state=123)
xgbod_cv_model.fit(X_scaled, y_train)

print("The following are the optimal parameters after cross validation:")
print(xgbod_cv_model.best_params_)

learning_rate = xgbod_cv_model.best_params_['learning_rate']
max_depth = xgbod_cv_model.best_params_['max_depth']
n_estimators = xgbod_cv_model.best_params_['n_estimators']
booster = xgbod_cv_model.best_params_['booster']
min_child_weight = xgbod_cv_model.best_params_['min_child_weight']
reg_alpha = xgbod_cv_model.best_params_['reg_alpha']
reg_lambda = xgbod_cv_model.best_params_['reg_lambda']

xgbod_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

xgbod_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

xgbod_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running XGBOD model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()

    xgbod_model = XGBOD(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                        booster=booster, min_child_weight=min_child_weight,
                        reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=123)
    xgbod_model.fit(X_scaled, y_train)
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = xgbod_model.predict(X_test_scaled)
    y_pred_proba = xgbod_model.predict_proba(X_test_scaled)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    xgbod_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    xgbod_metrics['precision'].append(precision_score(y_test, y_pred))
    xgbod_metrics['recall'].append(recall_score(y_test, y_pred))
    xgbod_metrics['f1'].append(f1_score(y_test, y_pred))
    xgbod_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    xgbod_metrics['training_time'].append(training_time)
    xgbod_metrics['pred_time'].append(pred_time)
    
    xgbod_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    xgbod_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    xgbod_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    xgbod_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    xgbod_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    xgbod_metrics_30['training_time'].append(training_time)
    xgbod_metrics_30['pred_time'].append(pred_time)
    
    xgbod_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    xgbod_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    xgbod_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    xgbod_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    xgbod_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    xgbod_metrics_70['training_time'].append(training_time)
    xgbod_metrics_70['pred_time'].append(pred_time)

xgbod_metrics_mean = {k:np.mean(v) for k, v in xgbod_metrics.items()}
print("XGBOD model mean metrics:")
print(xgbod_metrics_mean)

xgbod_metrics_30_mean = {k:np.mean(v) for k, v in xgbod_metrics_30.items()}
print("XGBOD model mean metrics with 30% threshold:")
print(xgbod_metrics_30_mean)

xgbod_metrics_70_mean = {k:np.mean(v) for k, v in xgbod_metrics_70.items()}
print("XGBOD model mean metrics with 70% threshold:")
print(xgbod_metrics_70_mean)

feawad_params = {
    'epochs': [10, 20, 50, 100],
    'batch_size': [16, 32, 64, 128],
    'lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    'act': ['ReLU', 'LeakyReLU', 'Sigmoid'],
    'bias': [True, False]
    }

print("Performing randomized Cross Validation on the FeaWAD model...")

keys, values = zip(*feawad_params.items())
param_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
random.seed(123)
random_params = random.choices(param_permutations, k=10)

param_scores = list()

for params in random_params:

    kf = KFold(n_splits=5, shuffle=False)
        
    fold_scores = list()

    for train_index, test_index in kf.split(X_scaled):
        train_fold_X = np.take(X_scaled, train_index, axis=0)
        test_fold_X = np.take(X_scaled, test_index, axis=0)
        train_fold_y = y_train.iloc[train_index].to_numpy()
        test_fold_y = y_train.iloc[test_index].to_numpy()

        feawad_model_fold = FeaWAD(**params)
        feawad_model_fold.fit(train_fold_X, train_fold_y)
        y_pred = feawad_model_fold.predict(test_fold_X)

        fold_score = roc_auc_score(test_fold_y, y_pred)

        fold_scores.append(fold_score)
    
    param_score = np.mean(fold_scores)
    param_scores.append(param_score)

best_param_index = np.argmax(param_scores)
best_params = random_params[best_param_index]

print("The following are the optimal parameters after cross validation:")
print(best_params)

epochs = best_params['epochs']
batch_size = best_params['batch_size']
lr = best_params['lr']
act = best_params['act']
bias = best_params['bias']

feawad_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

feawad_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

feawad_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running FeaWAD model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    feawad_model = FeaWAD(epochs=epochs, batch_size=batch_size, lr=lr, act=act, bias=bias)
    feawad_model.fit(X_scaled, y_train.to_numpy())
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred, y_pred_proba = feawad_model.predict(X_test_scaled, return_confidence=True)
    # y_pred_proba = feawad_model.predict_proba(X_test)[:,1]
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    feawad_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    feawad_metrics['precision'].append(precision_score(y_test, y_pred))
    feawad_metrics['recall'].append(recall_score(y_test, y_pred))
    feawad_metrics['f1'].append(f1_score(y_test, y_pred))
    feawad_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    feawad_metrics['training_time'].append(training_time)
    feawad_metrics['pred_time'].append(pred_time)
    
    feawad_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    feawad_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    feawad_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    feawad_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    feawad_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    feawad_metrics_30['training_time'].append(training_time)
    feawad_metrics_30['pred_time'].append(pred_time)
    
    feawad_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    feawad_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    feawad_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    feawad_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    feawad_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    feawad_metrics_70['training_time'].append(training_time)
    feawad_metrics_70['pred_time'].append(pred_time)
    
feawad_metrics_mean = {k:np.mean(v) for k, v in feawad_metrics.items()}
print("FeaWAD model mean metrics:")
print(feawad_metrics_mean)

feawad_metrics_30_mean = {k:np.mean(v) for k, v in feawad_metrics_30.items()}
print("FeaWAD model mean metrics with 30% threshold:")
print(feawad_metrics_30_mean)

feawad_metrics_70_mean = {k:np.mean(v) for k, v in feawad_metrics_70.items()}
print("FeaWAD model mean metrics with 70% threshold:")
print(feawad_metrics_70_mean)

deepsad_params = {
    'epochs': [10, 20, 50, 100],
    'batch_size': [16, 32, 64, 128],
    'lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    'act': ['ReLU', 'LeakyReLU', 'Sigmoid'],
    'bias': [True, False]
    }

print("Performing randomized Cross Validation on the DeepSAD model...")

keys, values = zip(*deepsad_params.items())
param_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
random.seed(123)
random_params = random.choices(param_permutations, k=10)

param_scores = list()

for params in random_params:

    kf = KFold(n_splits=5, shuffle=False)
        
    fold_scores = list()

    for train_index, test_index in kf.split(X_scaled):
        train_fold_X = np.take(X_scaled, train_index, axis=0)
        test_fold_X = np.take(X_scaled, test_index, axis=0)
        train_fold_y = y_train.iloc[train_index].to_numpy()
        test_fold_y = y_train.iloc[test_index].to_numpy()

        deepsad_model_fold = DeepSAD(**params)
        deepsad_model_fold.fit(train_fold_X, train_fold_y)
        y_pred = deepsad_model_fold.predict(test_fold_X)

        fold_score = roc_auc_score(test_fold_y, y_pred)

        fold_scores.append(fold_score)
    
    param_score = np.mean(fold_scores)
    param_scores.append(param_score)

best_param_index = np.argmax(param_scores)
best_params = random_params[best_param_index]

print("The following are the optimal parameters after cross validation:")
print(best_params)

epochs = best_params['epochs']
batch_size = best_params['batch_size']
lr = best_params['lr']
act = best_params['act']
bias = best_params['bias']

deepsad_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

deepsad_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

deepsad_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running DeepSAD model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    deepsad_model = DeepSAD(epochs=epochs, batch_size=batch_size, lr=lr, act=act, bias=bias)
    deepsad_model.fit(X_scaled, y_train.to_numpy())
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred, y_pred_proba = deepsad_model.predict(X_test_scaled, return_confidence=True)
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    deepsad_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    deepsad_metrics['precision'].append(precision_score(y_test, y_pred))
    deepsad_metrics['recall'].append(recall_score(y_test, y_pred))
    deepsad_metrics['f1'].append(f1_score(y_test, y_pred))
    deepsad_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    deepsad_metrics['training_time'].append(training_time)
    deepsad_metrics['pred_time'].append(pred_time)
    
    deepsad_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    deepsad_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    deepsad_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    deepsad_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    deepsad_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    deepsad_metrics_30['training_time'].append(training_time)
    deepsad_metrics_30['pred_time'].append(pred_time)
    
    deepsad_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    deepsad_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    deepsad_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    deepsad_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    deepsad_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    deepsad_metrics_70['training_time'].append(training_time)
    deepsad_metrics_70['pred_time'].append(pred_time)
    
deepsad_metrics_mean = {k:np.mean(v) for k, v in deepsad_metrics.items()}
print("DeepSAD model mean metrics:")
print(deepsad_metrics_mean)

deepsad_metrics_30_mean = {k:np.mean(v) for k, v in deepsad_metrics_30.items()}
print("DeepSAD model mean metrics with 30% threshold:")
print(deepsad_metrics_30_mean)

deepsad_metrics_70_mean = {k:np.mean(v) for k, v in deepsad_metrics_70.items()}
print("DeepSAD model mean metrics with 70% threshold:")
print(deepsad_metrics_70_mean)


prenet_params = {
    'epochs': [10, 20, 50, 100],
    'batch_size': [16, 32, 64, 128],
    'lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    'act': ['ReLU', 'LeakyReLU', 'Sigmoid'],
    'bias': [True, False]
    }

print("Performing randomized Cross Validation on the PReNet model...")

keys, values = zip(*prenet_params.items())
param_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
random.seed(123)
random_params = random.choices(param_permutations, k=10)

param_scores = list()

for params in random_params:

    kf = KFold(n_splits=5, shuffle=False)
        
    fold_scores = list()

    for train_index, test_index in kf.split(X_scaled):
        train_fold_X = np.take(X_scaled, train_index, axis=0)
        test_fold_X = np.take(X_scaled, test_index, axis=0)
        train_fold_y = y_train.iloc[train_index].to_numpy()
        test_fold_y = y_train.iloc[test_index].to_numpy()

        prenet_model_fold = PReNet(**params)
        prenet_model_fold.fit(train_fold_X, train_fold_y)
        y_pred = prenet_model_fold.predict(test_fold_X)

        fold_score = roc_auc_score(test_fold_y, y_pred)

        fold_scores.append(fold_score)
    
    param_score = np.mean(fold_scores)
    param_scores.append(param_score)

best_param_index = np.argmax(param_scores)
best_params = random_params[best_param_index]

print("The following are the optimal parameters after cross validation:")
print(best_params)

epochs = best_params['epochs']
batch_size = best_params['batch_size']
lr = best_params['lr']
act = best_params['act']
bias = best_params['bias']

prenet_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

prenet_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

prenet_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running PReNet model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    prenet_model = PReNet(epochs=epochs, batch_size=batch_size, lr=lr, act=act, bias=bias)
    prenet_model.fit(X_scaled, y_train.to_numpy())
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = prenet_model.predict(X_test_scaled)
    y_pred, y_pred_proba = prenet_model.predict(X_test_scaled, return_confidence=True)
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    prenet_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    prenet_metrics['precision'].append(precision_score(y_test, y_pred))
    prenet_metrics['recall'].append(recall_score(y_test, y_pred))
    prenet_metrics['f1'].append(f1_score(y_test, y_pred))
    prenet_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    prenet_metrics['training_time'].append(training_time)
    prenet_metrics['pred_time'].append(pred_time)
    
    prenet_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    prenet_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    prenet_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    prenet_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    prenet_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    prenet_metrics_30['training_time'].append(training_time)
    prenet_metrics_30['pred_time'].append(pred_time)
    
    prenet_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    prenet_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    prenet_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    prenet_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    prenet_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    prenet_metrics_70['training_time'].append(training_time)
    prenet_metrics_70['pred_time'].append(pred_time)
    
prenet_metrics_mean = {k:np.mean(v) for k, v in prenet_metrics.items()}
print("PReNet model mean metrics:")
print(prenet_metrics_mean)

prenet_metrics_30_mean = {k:np.mean(v) for k, v in prenet_metrics_30.items()}
print("PReNet model mean metrics with 30% threshold:")
print(prenet_metrics_30_mean)

prenet_metrics_70_mean = {k:np.mean(v) for k, v in prenet_metrics_70.items()}
print("PReNet model mean metrics with 70% threshold:")
print(prenet_metrics_70_mean)


devnet_params = {
    'epochs': [10, 20, 50, 100],
    'batch_size': [16, 32, 64, 128],
    'lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    'act': ['ReLU', 'LeakyReLU', 'Sigmoid'],
    'bias': [True, False]
    }

print("Performing randomized Cross Validation on the DevNet model...")

keys, values = zip(*devnet_params.items())
param_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
random.seed(123)
random_params = random.choices(param_permutations, k=10)

param_scores = list()

for params in random_params:

    kf = KFold(n_splits=5, shuffle=False)
        
    fold_scores = list()

    for train_index, test_index in kf.split(X_scaled):
        train_fold_X = np.take(X_scaled, train_index, axis=0)
        test_fold_X = np.take(X_scaled, test_index, axis=0)
        train_fold_y = y_train.iloc[train_index].to_numpy()
        test_fold_y = y_train.iloc[test_index].to_numpy()

        devnet_model_fold = DevNet(**params)
        devnet_model_fold.fit(train_fold_X, train_fold_y)
        y_pred = devnet_model_fold.predict(test_fold_X)

        fold_score = roc_auc_score(test_fold_y, y_pred)

        fold_scores.append(fold_score)
    
    param_score = np.mean(fold_scores)
    param_scores.append(param_score)

best_param_index = np.argmax(param_scores)
best_params = random_params[best_param_index]

print("The following are the optimal parameters after cross validation:")
print(best_params)

epochs = best_params['epochs']
batch_size = best_params['batch_size']
lr = best_params['lr']
act = best_params['act']
bias = best_params['bias']

devnet_metrics = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

devnet_metrics_30 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

devnet_metrics_70 = {
        'accuracy': list(),
        'precision': list(),
        'recall': list(),
        'f1': list(),
        'roc_auc': list(),
        'training_time': list(),
        'pred_time': list()
        }

for i in range(10):
    print("Running DevNet model... - run #" + str(i+1))
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    
    devnet_model = DevNet(epochs=epochs, batch_size=batch_size, lr=lr, act=act, bias=bias)
    devnet_model.fit(X_scaled, y_train.to_numpy())
    
    end_time = time.time()
    
    training_time = end_time - start_time
    
    start_time = time.time()
    
    y_pred = devnet_model.predict(X_test_scaled)
    y_pred, y_pred_proba = devnet_model.predict(X_test_scaled, return_confidence=True)
    
    end_time = time.time()
    
    pred_time = end_time - start_time
    
    y_pred_30 = (y_pred_proba >= 0.3).astype(int)
    y_pred_70 = (y_pred_proba >= 0.7).astype(int)
    
    devnet_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    devnet_metrics['precision'].append(precision_score(y_test, y_pred))
    devnet_metrics['recall'].append(recall_score(y_test, y_pred))
    devnet_metrics['f1'].append(f1_score(y_test, y_pred))
    devnet_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred))
    devnet_metrics['training_time'].append(training_time)
    devnet_metrics['pred_time'].append(pred_time)
    
    devnet_metrics_30['accuracy'].append(accuracy_score(y_test, y_pred_30))
    devnet_metrics_30['precision'].append(precision_score(y_test, y_pred_30))
    devnet_metrics_30['recall'].append(recall_score(y_test, y_pred_30))
    devnet_metrics_30['f1'].append(f1_score(y_test, y_pred_30))
    devnet_metrics_30['roc_auc'].append(roc_auc_score(y_test, y_pred_30))
    devnet_metrics_30['training_time'].append(training_time)
    devnet_metrics_30['pred_time'].append(pred_time)
    
    devnet_metrics_70['accuracy'].append(accuracy_score(y_test, y_pred_70))
    devnet_metrics_70['precision'].append(precision_score(y_test, y_pred_70))
    devnet_metrics_70['recall'].append(recall_score(y_test, y_pred_70))
    devnet_metrics_70['f1'].append(f1_score(y_test, y_pred_70))
    devnet_metrics_70['roc_auc'].append(roc_auc_score(y_test, y_pred_70))
    devnet_metrics_70['training_time'].append(training_time)
    devnet_metrics_70['pred_time'].append(pred_time)
    
devnet_metrics_mean = {k:np.mean(v) for k, v in devnet_metrics.items()}
print("DevNet model mean metrics:")
print(devnet_metrics_mean)

devnet_metrics_30_mean = {k:np.mean(v) for k, v in devnet_metrics_30.items()}
print("DevNet model mean metrics with 30% threshold:")
print(devnet_metrics_30_mean)

devnet_metrics_70_mean = {k:np.mean(v) for k, v in devnet_metrics_70.items()}
print("DevNet model mean metrics with 70% threshold:")
print(devnet_metrics_70_mean)
