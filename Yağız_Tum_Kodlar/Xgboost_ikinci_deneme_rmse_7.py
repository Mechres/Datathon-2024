import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import BaseEstimator, TransformerMixin

# Veri setlerini yükle
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# Hedef değişkeni ayır ve temizle
y = train['Degerlendirme Puani']
y = y.fillna(y.mean())
y = y.replace([np.inf, -np.inf], np.nan).dropna()

X = train.drop(['Degerlendirme Puani', 'id'], axis=1)
X = X.loc[y.index]


class DateFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.date_columns:
            if col in X_.columns:
                X_[col] = pd.to_datetime(X_[col], errors='coerce')
                X_[f'{col}_year'] = X_[col].dt.year
                X_[f'{col}_month'] = X_[col].dt.month
                X_[f'{col}_day'] = X_[col].dt.day
                X_[f'{col}_dayofweek'] = X_[col].dt.dayofweek
                X_[f'{col}_quarter'] = X_[col].dt.quarter
                X_[f'{col}_is_weekend'] = X_[col].dt.dayofweek.isin([5, 6]).astype(int)
                X_ = X_.drop(col, axis=1)

        # Yaş hesapla
        if 'Dogum Tarihi_year' in X_.columns:
            current_year = pd.Timestamp.now().year
            X_['Age'] = current_year - X_['Dogum Tarihi_year']

        return X_


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.freq_dict = {}

    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                self.freq_dict[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.columns:
            if col in X_.columns and col in self.freq_dict:
                X_[f'{col}_freq'] = X_[col].map(self.freq_dict[col]).fillna(0)
        return X_


def get_feature_types(X):
    date_columns = ['Dogum Tarihi']
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'Dogum Tarihi' in categorical_features:
        categorical_features.remove('Dogum Tarihi')

    return date_columns, categorical_features, numeric_features


date_columns, categorical_features, numeric_features = get_feature_types(X)

preprocessor = Pipeline([
    ('date_transformer', DateFeatureTransformer(date_columns=date_columns)),
    ('freq_encoder', FrequencyEncoder(columns=categorical_features)),
    ('column_transformer', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ], remainder='passthrough')),
    ('final_imputer', SimpleImputer(strategy='mean'))
])

# XGBoost modeli
xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_regression, k=20)),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# LightGBM modeli
lgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_regression, k=20)),
    ('regressor', LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Ensemble model
ensemble_model = VotingRegressor([
    ('xgb', xgb_model),
    ('lgb', lgb_model)
])

# Eğitim ve test setlerini ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
ensemble_model.fit(X_train, y_train)

# Validation seti üzerinde tahmin yap ve performansı değerlendir
val_predictions = ensemble_model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = root_mean_squared_error(y_val, val_predictions)

print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")

# Test seti üzerinde tahmin yap
test_predictions = ensemble_model.predict(test.drop('id', axis=1))

# Sonuçları kaydet
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Tahminler 'submission.csv' dosyasına kaydedildi.")