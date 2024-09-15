import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from catboost import CatBoostRegressor

# Eğitim ve test veri setlerini yükle
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# Hedef değişkeni ayır ve temizle
y = train['Degerlendirme Puani'].fillna(train['Degerlendirme Puani'].mean())
y = y.replace([np.inf, -np.inf], np.nan).dropna()

# Girdi değişkenlerini seç
X = train.drop(['Degerlendirme Puani', 'id'], axis=1)

# Kategorik ve sayısal kolonları belirle
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Tarih sütunlarını dönüştür
date_columns = ['Dogum Tarihi']
for col in date_columns:
    X[col] = pd.to_datetime(X[col], errors='coerce')
    X[f'{col}_year'] = X[col].dt.year
    X[f'{col}_month'] = X[col].dt.month
    X = X.drop(col, axis=1)
    test[col] = pd.to_datetime(test[col], errors='coerce')
    test[f'{col}_year'] = test[col].dt.year
    test[f'{col}_month'] = test[col].dt.month
    test = test.drop(col, axis=1)

# Kategorik ve sayısal kolonları güncelle
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Veri ön işleme pipeline'ı oluştur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_columns),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_columns)
    ])

# Model pipeline'ı oluştur
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hiperparametre arama alanını küçült
param_grid = {
    'regressor__n_estimators': [50, 100],  # Daha az ağaç sayısı
    'regressor__max_depth': [None, 10],  # Maksimum derinliği sınırla
}

# MSE skoru kullanarak en iyi parametreleri bulmak için GridSearchCV kullan
grid_search = GridSearchCV(
    model, param_grid, cv=3,  # Çapraz doğrulama sayısını azalt
    scoring=make_scorer(mean_squared_error, greater_is_better=False), 
    verbose=2, n_jobs=1  # Paralel işlem sayısını 1'e indir
)

# Eğitim ve test setlerini ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim verisinin bir kısmını kullanarak bellek kullanımını azalt
X_train_sample = X_train.sample(frac=0.3, random_state=42)  # Verinin %30'unu kullan
y_train_sample = y_train[X_train_sample.index]

# Modeli eğit
grid_search.fit(X_train_sample, y_train_sample)

# En iyi modeli elde et
best_model = grid_search.best_estimator_

# Validation seti üzerinde tahmin yap ve performansı değerlendir
val_predictions = best_model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")

# Test seti üzerinde tahmin yap
test_predictions = best_model.predict(test.drop('id', axis=1))

# Sonuçları kaydet
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Tahminler 'submission.csv' dosyasına kaydedildi.")
