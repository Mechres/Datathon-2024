import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

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
X_train_full = train_df.drop(['Degerlendirme Puani', 'id', 'Dogum Tarihi'], axis=1)
y_train_full = train_df['Degerlendirme Puani']
X_test = test_df.drop(['id', 'Dogum Tarihi'], axis=1)

# NaN değerleri kontrol et ve raporla
print("NaN değerler kontrol ediliyor...")
print("Eğitim verisinde NaN değerler:")
print(X_train_full.isna().sum())
print("\nHedef değişkende NaN değerler:")
print(y_train_full.isna().sum())

# NaN değerleri olan satırları kaldır (eğer hala varsa)
mask = ~np.isnan(y_train_full)
X_train_full = X_train_full[mask]
y_train_full = y_train_full[mask]

# Eğitim setini train/test olarak ayır (gerçek test seti olmadığı için)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# MODELLERİ EĞİT VE TEST ET
# 1. Random Forest
print("Random Forest modeli eğitiliyor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
print(f"Random Forest RMSE: {rf_rmse:.4f}")

# 2. XGBoost
print("XGBoost modeli eğitiliyor...")
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")

# 3. CatBoost
print("CatBoost modeli eğitiliyor...")
catboost_model = CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
catboost_model.fit(X_train, y_train)
catboost_pred = catboost_model.predict(X_val)
catboost_rmse = np.sqrt(mean_squared_error(y_val, catboost_pred))
print(f"CatBoost RMSE: {catboost_rmse:.4f}")

# Test seti ile tahmin yap ve dosyaya kaydet
def save_predictions(model_name, predictions):
    output_df = test_df.copy()
    output_df['Degerlendirme Puani'] = predictions
    output_df['id'] = output_df['id'].astype(int)  # 'id' sütunu int formatına dönüştürülüyor
    output_file = f'submission_{model_name}.csv'
    output_df[['id', 'Degerlendirme Puani']].to_csv(output_file, index=False)
    print(f"Tahminler '{output_file}' dosyasına kaydedildi.")

# Her modelin sonuçlarını kaydet
rf_pred_test = rf_model.predict(X_test)
save_predictions('random_forest', rf_pred_test)

xgb_pred_test = xgb_model.predict(X_test)
save_predictions('xgboost', xgb_pred_test)

catboost_pred_test = catboost_model.predict(X_test)
save_predictions('catboost', catboost_pred_test)
