import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# Veri setlerini yükle
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# Hedef değişkeni ayır ve temizle
y = train['Degerlendirme Puani']
y = y.fillna(y.mean())
y = y.replace([np.inf, -np.inf], np.nan).dropna()

X = train.drop(['Degerlendirme Puani', 'id'], axis=1)
X = X.loc[y.index]

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

# Kategorik sütunları 'str' türüne dönüştür
X[categorical_columns] = X[categorical_columns].astype(str)
test[categorical_columns] = test[categorical_columns].astype(str)

# Veri ön işleme pipeline'ı oluştur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_columns),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))  # Sparse formatta saklayın
        ]), categorical_columns)
    ])

# Model pipeline'ı oluştur
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Eğitim ve test setlerini ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model.fit(X_train, y_train)

# Validation seti üzerinde tahmin yap ve performansı değerlendir
val_predictions = model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = root_mean_squared_error(y_val, val_predictions)

print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")

# Test seti üzerinde tahmin yap
test_predictions = model.predict(test.drop('id', axis=1))

# Sonuçları kaydet
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Tahminler 'submission.csv' dosyasına kaydedildi.")