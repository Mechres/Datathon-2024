import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


# Çok Uzun sürüyor, 3 saat sonunda bitmedi İPTAL!

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

class InteractionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, interaction_only=False):
        self.degree = degree
        self.interaction_only = interaction_only

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        poly = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only, include_bias=False)
        return poly.fit_transform(X)


# Veri yükleme
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

y = train['Degerlendirme Puani']
y = y.fillna(y.mean())
y = y.replace([np.inf, -np.inf], np.nan).dropna()

X = train.drop(['Degerlendirme Puani', 'id'], axis=1)
X = X.loc[y.index]

date_columns, categorical_features, numeric_features = get_feature_types(X)

preprocessor = Pipeline([
    ('date_transformer', DateFeatureTransformer(date_columns=date_columns)),
    ('freq_encoder', FrequencyEncoder(columns=categorical_features)),
    ('column_transformer', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('interaction', InteractionFeatures(degree=2, interaction_only=True))
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]), categorical_features)
    ], remainder='passthrough')),
    ('final_imputer', SimpleImputer(strategy='mean')),
    ('feature_selection', SelectKBest(f_regression, k=50))
])

# Model tanımlamaları
xgb_model = XGBRegressor(random_state=42)
lgb_model = LGBMRegressor(random_state=42)
catboost_model = CatBoostRegressor(random_state=42, verbose=0)
rf_model = RandomForestRegressor(random_state=42)

# Stacking Ensemble
stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('catboost', catboost_model),
        ('rf', rf_model)
    ],
    final_estimator=Lasso(),
    cv=5
)

# Final pipeline
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', stacking_model)
])

# Hiperparametre arama uzayı
param_distributions = {
    'stacking__xgb__n_estimators': [100, 200, 300],
    'stacking__xgb__learning_rate': [0.01, 0.1, 0.3],
    'stacking__xgb__max_depth': [3, 5, 7],
    'stacking__lgb__n_estimators': [100, 200, 300],
    'stacking__lgb__learning_rate': [0.01, 0.1, 0.3],
    'stacking__lgb__num_leaves': [31, 63, 127],
    'stacking__catboost__iterations': [100, 200, 300],
    'stacking__catboost__learning_rate': [0.01, 0.1, 0.3],
    'stacking__catboost__depth': [4, 6, 8],
    'stacking__rf__n_estimators': [100, 200, 300],
    'stacking__rf__max_depth': [None, 10, 20],
    'stacking__final_estimator__alpha': [0.1, 1.0, 10.0]
}

# Randomized Search CV
random_search = RandomizedSearchCV(
    final_model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Model eğitimi
print("Model eğitimi başlıyor...")
random_search.fit(X, y)

# En iyi model ve parametreler
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_score = -random_search.best_score_  # Negatif skorun tersini al

print(f"En iyi RMSE: {best_score}")
print("En iyi parametreler:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Test seti üzerinde tahmin
test_predictions = best_model.predict(test.drop('id', axis=1))

# Sonuçları kaydet
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Tahminler 'submission.csv' dosyasına kaydedildi.")

# K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
cv_rmse = -cv_scores

print(f"Cross-Validation RMSE: {cv_rmse.mean()} (+/- {cv_rmse.std() * 2})")