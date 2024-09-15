import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import warnings

warnings.filterwarnings('ignore')


# Geliştirilmiş Özel One-Hot Encoder sınıfı
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = None

    def fit(self, X, y=None):
        X_temp = pd.DataFrame(X)
        self.encoder = pd.get_dummies(X_temp).columns
        return self

    def transform(self, X):
        X_temp = pd.DataFrame(X)
        X_encoded = pd.get_dummies(X_temp)
        return X_encoded.reindex(columns=self.encoder, fill_value=0)


# Veri yükleme ve ön işleme
print("Veri yükleniyor...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

y = train['Degerlendirme Puani'].fillna(train['Degerlendirme Puani'].mean())
X = train.drop(['Degerlendirme Puani', 'id'], axis=1)

# Basit özellik mühendisliği
print("Özellik mühendisliği yapılıyor...")
X['Dogum Tarihi'] = pd.to_datetime(X['Dogum Tarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
current_date = pd.Timestamp.now()
X['Age'] = (current_date.year - X['Dogum Tarihi'].dt.year)
X.loc[X['Dogum Tarihi'].dt.month > current_date.month, 'Age'] -= 1
X = X.drop('Dogum Tarihi', axis=1)

# Kategorik ve sayısal özellikleri ayırma
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Veri önişleme pipeline'ı
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', CustomOneHotEncoder())
    ]), categorical_features)
])

# XGBoost model
xgb_model = XGBRegressor(random_state=42, n_jobs=-1)

# Final pipeline
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_regression, k=50)),
    ('regressor', xgb_model)
])

# Hiperparametre arama uzayı
param_distributions = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 5, 7],
    'regressor__subsample': [0.8, 0.9, 1.0],
    'regressor__colsample_bytree': [0.8, 0.9, 1.0]
}

# Randomized Search CV
print("Hiperparametre optimizasyonu başlıyor...")
random_search = RandomizedSearchCV(
    final_model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

# Model eğitimi
print("Model eğitimi başlıyor...")
try:
    random_search.fit(X, y)

    # En iyi model ve parametreler
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = -random_search.best_score_

    print(f"En iyi RMSE: {best_score}")
    print("En iyi parametreler:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Test seti üzerinde tahmin
    print("Test seti üzerinde tahmin yapılıyor...")
    test_id = test['id']  # ID'leri sakla
    test['Dogum Tarihi'] = pd.to_datetime(test['Dogum Tarihi'], format='%d.%m.%Y %H:%M', errors='coerce')
    test['Age'] = (current_date.year - test['Dogum Tarihi'].dt.year)
    test.loc[test['Dogum Tarihi'].dt.month > current_date.month, 'Age'] -= 1
    test = test.drop(['Dogum Tarihi', 'id'], axis=1)

    test_predictions = best_model.predict(test)

    # NaN değerleri kontrol et
    if np.isnan(test_predictions).any():
        print("Uyarı: Tahminlerde NaN değerler var!")
        test_predictions = np.nan_to_num(test_predictions, nan=y.mean())

    # Sonuçları kaydet
    submission = pd.DataFrame({
        'id': test_id,
        'Degerlendirme Puani': test_predictions
    })
    submission.to_csv('submission.csv', index=False)

    print("Tahminler 'submission.csv' dosyasına kaydedildi.")

    # Cross-Validation
    print("Cross-validation yapılıyor...")
    cv_scores = cross_val_score(best_model, X, y, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_rmse = -cv_scores

    print(f"Cross-Validation RMSE: {cv_rmse.mean()} (+/- {cv_rmse.std() * 2})")

except Exception as e:
    print(f"Bir hata oluştu: {e}")
    import traceback

    print(traceback.format_exc())