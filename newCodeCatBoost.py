import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Eğitim ve test verilerini yükle
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test_x.csv')

# Eğitim verilerinden hedef değişkeni ve özellikleri ayır
X_train = train_data.drop(['Degerlendirme Puani', 'id'], axis=1)
y_train = train_data['Degerlendirme Puani']

# Test verisindeki id sütununu ayır
test_ids = test_data['id']
X_test = test_data.drop(['id'], axis=1)

# Hedef değişkendeki NaN değerleri ortalama ile doldur
y_train = y_train.fillna(y_train.mean())

# Kategorik özelliklerin listesini al
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Kategorik sütunlardaki NaN değerleri 'missing' olarak doldur
X_train[categorical_features] = X_train[categorical_features].fillna('missing')
X_test[categorical_features] = X_test[categorical_features].fillna('missing')

# Tarih sütunlarını işleyin (varsa)
date_columns = ['Dogum Tarihi']
for col in date_columns:
    if col in X_train.columns:
        X_train[col] = pd.to_datetime(X_train[col], errors='coerce')
        X_train[f'{col}_year'] = X_train[col].dt.year
        X_train[f'{col}_month'] = X_train[col].dt.month
        X_train[f'{col}_day'] = X_train[col].dt.day
        X_train[f'{col}_weekday'] = X_train[col].dt.weekday
        X_train[f'{col}_is_weekend'] = X_train[col].dt.weekday >= 5
        X_train = X_train.drop(col, axis=1)

        X_test[col] = pd.to_datetime(X_test[col], errors='coerce')
        X_test[f'{col}_year'] = X_test[col].dt.year
        X_test[f'{col}_month'] = X_test[col].dt.month
        X_test[f'{col}_day'] = X_test[col].dt.day
        X_test[f'{col}_weekday'] = X_test[col].dt.weekday
        X_test[f'{col}_is_weekend'] = X_test[col].dt.weekday >= 5
        X_test = X_test.drop(col, axis=1)

# Kategorik ve sayısal kolonları güncelle
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Eğitim ve doğrulama setlerine ayır
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# CatBoostRegressor için hiperparametre ızgarasını tanımla
param_grid = {
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7],
    'random_seed': [42],
    'early_stopping_rounds': [50]  # Early stopping ekledik
}

# CatBoost modelini tanımla
model = CatBoostRegressor(cat_features=categorical_features, verbose=100)

# GridSearchCV ile hiperparametre optimizasyonu yap
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)

# Modeli eğitim verileri ile eğit
grid_search.fit(X_train_split, y_train_split, eval_set=(X_val, y_val), early_stopping_rounds=50)

# En iyi modeli al
best_model = grid_search.best_estimator_

# En iyi parametreleri yazdır
print(f"En iyi parametreler: {grid_search.best_params_}")

# Validation seti üzerinde tahmin yap ve performansı değerlendir
val_predictions = best_model.predict(X_val)

# MSE ve RMSE hesapla
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)

# MSE ve RMSE ekrana yazdır
print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")

# Test setini tahmin et
test_predictions = best_model.predict(X_test)

# Sonuçları kaydet
submission = pd.DataFrame({
    'id': test_ids,
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Tahminler 'submission.csv' dosyasına kaydedildi.")

# Modelin önem sırasını yazdır
feature_importances = best_model.get_feature_importance(prettified=True)
print("Özellik Önem Sırası:")
print(feature_importances)
