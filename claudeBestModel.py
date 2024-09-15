import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Veri setlerini yükle
print("Veri setleri yükleniyor...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_x.csv')  # Test dosyası adı güncellendi

# Eğitim ve test verilerini birleştir (etiket kodlama için)
print("Veriler birleştiriliyor ve ön işleme yapılıyor...")
train_df['is_train'] = 1
test_df['is_train'] = 0
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Tarih sütunlarını datetime'a çevir ve yıl bilgisini al
date_columns = ['Dogum Tarihi']
for col in date_columns:
    combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')
    combined_df[f'{col}_Year'] = combined_df[col].dt.year

# Kategorik değişkenleri kodla
le = LabelEncoder()
categorical_columns = combined_df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))

# Sayısal değişkenlerdeki eksik değerleri doldur
numeric_columns = combined_df.select_dtypes(include=['int64', 'float64']).columns
imputer = SimpleImputer(strategy='mean')
combined_df[numeric_columns] = imputer.fit_transform(combined_df[numeric_columns])

# Kategorik değişkenlerdeki eksik değerleri doldur
categorical_columns = combined_df.select_dtypes(include=['object']).columns
combined_df[categorical_columns] = combined_df[categorical_columns].fillna('Unknown')

# Eğitim ve test verilerini ayır
train_df = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1)
test_df = combined_df[combined_df['is_train'] == 0].drop(['is_train', 'Degerlendirme Puani'], axis=1)

# Özellikleri ve hedef değişkeni ayır
X_train = train_df.drop(['Degerlendirme Puani', 'id', 'Dogum Tarihi'], axis=1)
y_train = train_df['Degerlendirme Puani']
X_test = test_df.drop(['id', 'Dogum Tarihi'], axis=1)

# NaN değerleri kontrol et ve raporla
print("NaN değerler kontrol ediliyor...")
print("Eğitim verisinde NaN değerler:")
print(X_train.isna().sum())
print("\nHedef değişkende NaN değerler:")
print(y_train.isna().sum())

# NaN değerleri olan satırları kaldır (eğer hala varsa)
mask = ~np.isnan(y_train)
X_train = X_train[mask]
y_train = y_train[mask]

# MODELLERİ EĞİT VE KARŞILAŞTIR
# 1. Random Forest
print("Random Forest modeli eğitiliyor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred_test = rf_model.predict(X_test)

# 2. XGBoost
print("XGBoost modeli eğitiliyor...")
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred_train = xgb_model.predict(X_train)
xgb_pred_test = xgb_model.predict(X_test)

# 3. CatBoost
print("CatBoost modeli eğitiliyor...")
catboost_model = CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
catboost_model.fit(X_train, y_train)
catboost_pred_train = catboost_model.predict(X_train)
catboost_pred_test = catboost_model.predict(X_test)

# Performans karşılaştırması (eğitim setinde)
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

print("\nModel Sonuçları (Eğitim Seti):")

# Random Forest sonuçları
rf_mse, rf_rmse, rf_r2 = evaluate_model(y_train, rf_pred_train)
print(f"Random Forest -> MSE: {rf_mse}, RMSE: {rf_rmse}, R²: {rf_r2}")

# XGBoost sonuçları
xgb_mse, xgb_rmse, xgb_r2 = evaluate_model(y_train, xgb_pred_train)
print(f"XGBoost -> MSE: {xgb_mse}, RMSE: {xgb_rmse}, R²: {xgb_r2}")

# CatBoost sonuçları
catboost_mse, catboost_rmse, catboost_r2 = evaluate_model(y_train, catboost_pred_train)
print(f"CatBoost -> MSE: {catboost_mse}, RMSE: {catboost_rmse}, R²: {catboost_r2}")

# Tahminleri farklı dosyalara kaydet
def save_predictions(model_name, predictions):
    output_df = test_df.copy()
    output_df['Degerlendirme Puani'] = predictions
    output_df['id'] = output_df['id'].astype(int)  # 'id' sütunu int formatına dönüştürülüyor
    output_file = f'submission_{model_name}.csv'
    output_df[['id', 'Degerlendirme Puani']].to_csv(output_file, index=False)
    print(f"Tahminler '{output_file}' dosyasına kaydedildi.")

# Her modelin sonuçlarını kaydet
save_predictions('random_forest', rf_pred_test)
save_predictions('xgboost', xgb_pred_test)
save_predictions('catboost', catboost_pred_test)
