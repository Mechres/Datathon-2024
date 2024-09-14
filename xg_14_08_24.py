import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

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
    X[f'{col}_weekday'] = X[col].dt.weekday  # Haftanın gününü ekle
    X = X.drop(col, axis=1)

    test[col] = pd.to_datetime(test[col], errors='coerce')
    test[f'{col}_year'] = test[col].dt.year
    test[f'{col}_month'] = test[col].dt.month
    test[f'{col}_weekday'] = test[col].dt.weekday  # Haftanın gününü ekle
    test = test.drop(col, axis=1)

# Kategorik ve sayısal kolonları güncelle
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Kategorik sütunları 'str' türüne dönüştür
X[categorical_columns] = X[categorical_columns].astype(str)
test[categorical_columns] = test[categorical_columns].astype(str)

# Veri ön işleme pipeline'ı oluştur
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Sparse formatta saklamayın
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Kullanılabilir modeller
models = {
    "1": ("XGBoost", XGBRegressor(random_state=42, tree_method='gpu_hist')),  # GPU hızlandırması
    "2": ("Random Forest", RandomForestRegressor(random_state=42)),
    "3": ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    "4": ("SVR", SVR())
}

# Kullanıcıdan model seçimi al
print("Kullanılabilir Modeller:")
for key, value in models.items():
    print(f"{key}: {value[0]}")

selected_models = input("Lütfen kullanmak istediğiniz model numaralarını virgülle ayırarak girin (veya 'all' yazın): ")

if selected_models.lower() == "all":
    selected_models = list(models.keys())
else:
    selected_models = selected_models.split(",")

# Seçilen modeller için döngü
for model_num in selected_models:
    if model_num not in models:
        print(f"Geçersiz model numarası: {model_num}, atlanıyor.")
        continue

    model_name, regressor = models[model_num]

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    # Eğitim ve test setlerini ayır
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hiperparametre Ayarlaması (isteğe bağlı)
    param_grid = {
        # Modelinize göre uygun hiperparametreleri ayarlayın
        'regressor__n_estimators': [100, 200, 300],  # XGBoost veya GradientBoosting için
        'regressor__learning_rate': [0.01, 0.1, 0.2],  # XGBoost veya GradientBoosting için
        'regressor__max_depth': [3, 5, 7],  # XGBoost veya GradientBoosting veya RandomForest için
        # 'regressor__C': [0.1, 1, 10],  # SVR için
        # 'regressor__gamma': ['scale', 'auto']  # SVR için
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Model adı ile sonuçları yazdır
    print(f"\n{model_name} Sonuçları:")
    print("En iyi parametreler:", grid_search.best_params_)
    print("En iyi skor:", -grid_search.best_score_)

    # En iyi modeli kullanarak tahmin yapın
    best_model = grid_search.best_estimator_

    # Validation seti üzerinde tahmin yap ve performansı değerlendir
    val_predictions = best_model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    rmse = root_mean_squared_error(y_val, val_predictions)

    print(f"Validation MSE: {mse}")
    print(f"Validation RMSE: {rmse}")

    # Test seti üzerinde tahmin yap
    test_predictions = best_model.predict(test.drop('id', axis=1))

    # Sonuçları kaydet
    submission = pd.DataFrame({
        'id': test['id'],
        'Degerlendirme Puani': test_predictions
    })
    submission.to_csv(f'submission_{model_name}.csv', index=False)

    print(f"{model_name} tahminleri 'submission_{model_name}.csv' dosyasına kaydedildi.")