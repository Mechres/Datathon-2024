# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 05:38:12 2024

@author: TDM
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

train_df = pd.read_csv("train.csv", low_memory=False)
test_df = pd.read_csv("test_x.csv", low_memory=False)

train_df.shape, test_df.shape, len(train_df), len(test_df)

train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

missing_train = train_df.isna().mean() * 100
missing_test = test_df.isna().mean() * 100

missing_values = train_df.isnull().mean() * 100
missing_values = missing_values[missing_values > 0]
missing_values = missing_values.sort_values(ascending=False)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


missing_threshold = 0.75

high_missing_columns = train_df.columns[train_df.isnull().mean() > missing_threshold]
train_df = train_df.drop(columns=high_missing_columns)
test_df = test_df.drop(columns=high_missing_columns)

target = 'Degerlendirme Puani'

for column in train_df.columns:
    if train_df[column].isnull().any():
        if train_df[column].dtype == 'object':
            mode_value = train_df[column].mode()[0]
            train_df[column] = train_df[column].fillna(mode_value)
            if column != target:
                test_df[column] = test_df[column].fillna(mode_value)
        else:
            median_value = train_df[column].median()
            train_df[column] = train_df[column].fillna(median_value)
            if column != target:
                test_df[column] = test_df[column].fillna(median_value)

from sklearn.impute import KNNImputer
import pandas as pd


def knn_impute(df, n_neighbors=5):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=df_encoded.columns)
    for col in df.select_dtypes(include='object').columns:
        df_imputed[col] = df_imputed[col].round().astype(int).map(
            dict(enumerate(df[col].astype('category').cat.categories)))
    return df_imputed


df_train_imputed = knn_impute(train_df, n_neighbors=5)
df_test_imputed = knn_impute(test_df, n_neighbors=5)

# Check the length of the imputed datasets
print(f"Length of train_df after imputation: {len(df_train_imputed)}")
print(f"Length of test_df after imputation: {len(df_test_imputed)}")

cat_cols_train = df_train_imputed.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'Degerlendirme Puani']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df_train_imputed[cat_cols_train] = ordinal_encoder.fit_transform(df_train_imputed[cat_cols_train].astype(str))
df_test_imputed[cat_cols_train] = ordinal_encoder.transform(df_test_imputed[cat_cols_train].astype(str))

train_df = df_train_imputed
test_df = df_test_imputed

le = LabelEncoder()
train_df['Degerlendirme Puani'] = le.fit_transform(train_df['Degerlendirme Puani'])

y = train_df['Degerlendirme Puani']
X = train_df.drop(['Degerlendirme Puani'], axis=1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import mean_squared_error
import numpy as np


def rmse_metric(y_pred, dmatrix):
    y_true = dmatrix.get_label()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return 'rmse', rmse


from xgboost import XGBRegressor

model = XGBRegressor(
    alpha=0.2,
    reg_lambda=0.2,
    subsample=0.8,
    colsample_bytree=0.6,
    learning_rate=0.05,
    objective='reg:squarederror',
    max_depth=14,
    min_child_weight=7,
    gamma=1e-6,
    n_estimators=1000,
    eval_metric='rmse',
    early_stopping_rounds=10
)

XGB = model.fit(
    train_X,
    train_y,
    eval_set=[(test_X, test_y)],
    verbose=True
)

y_pred = model.predict(test_X)

