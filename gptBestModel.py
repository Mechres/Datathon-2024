import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Eğitim ve test verilerini yükle
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test_x.csv')

# Eğitim verilerinden hedef değişkeni ve özellikleri ayır
X_train = train_data.drop(['Degerlendirme Puani', 'id'], axis=1)
y_train = train_data['Degerlendirme Puani']

# Test verisindeki id sütununu ayır
test_ids = test_data['id']
X_test = test_data.drop(['id'], axis=1)

# NaN ve sonsuz değerleri doldur
y_train = y_train.fillna(y_train.mean())
y_train.replace([np.inf, -np.inf], y_train.mean(), inplace=True)

# Eğitim ve doğrulama setlerine ayır
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# NaN ve sonsuz değerleri doldur (validation seti için)
y_val = y_val.fillna(y_val.mean())
y_val.replace([np.inf, -np.inf], y_val.mean(), inplace=True)

# XGBoost modelini tanımlayın
model = XGBRegressor(random_state=42, tree_method='hist', enable_categorical=True)

# Modeli eğitim verisiyle eğitin
model.fit(X_train_split, y_train_split)

# Validation seti üzerinde tahmin yap ve performansı değerlendir
val_predictions = model.predict(X_val)

# MSE ve RMSE hesapla
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)

print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")

# Test setini tahmin et
test_predictions = model.predict(X_test)

# Sonuçları kaydet
submission = pd.DataFrame({
    'id': test_ids,
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Tahminler 'submission.csv' dosyasına kaydedildi.")
