import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# Separate the target variable and clean it
y = train['Degerlendirme Puani']
y = y.fillna(y.mean()).replace([np.inf, -np.inf], np.nan).dropna()

X = train.drop(['Degerlendirme Puani', 'id'], axis=1)
X = X.loc[y.index]

# Convert date columns
date_columns = ['Dogum Tarihi']
for col in date_columns:
    X[col] = pd.to_datetime(X[col], errors='coerce')
    X[f'{col}_year'] = X[col].dt.year
    X[f'{col}_month'] = X[col].dt.month
    X[f'{col}_day'] = X[col].dt.day
    X[f'{col}_quarter'] = X[col].dt.quarter
    X = X.drop(col, axis=1)
    test[col] = pd.to_datetime(test[col], errors='coerce')
    test[f'{col}_year'] = test[col].dt.year
    test[f'{col}_month'] = test[col].dt.month
    test[f'{col}_day'] = test[col].dt.day
    test[f'{col}_quarter'] = test[col].dt.quarter
    test = test.drop(col, axis=1)

# Update categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Convert categorical columns to 'str' type
X[categorical_columns] = X[categorical_columns].astype(str)
test[categorical_columns] = test[categorical_columns].astype(str)

# Data preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', MinMaxScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)])

# Model pipeline
xgb_model = XGBRegressor(random_state=42)
lgbm_model = LGBMRegressor(random_state=42)
catboost_model = CatBoostRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

estimators = [
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('catboost', catboost_model),
    ('gb', gb_model)]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=xgb_model)

# Create the pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', stacking_model)
])

# Parameters for GridSearchCV
params = {
    'regressor__final_estimator__n_estimators': [100, 200, 500],
    'regressor__final_estimator__learning_rate': [0.01, 0.1, 0.3],
    'regressor__final_estimator__max_depth': [3, 5, 7],
    'regressor__final_estimator__subsample': [0.8, 1.0],
    'regressor__final_estimator__colsample_bytree': [0.8, 1.0]
}

# Hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform hyperparameter tuning and fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the validation set and evaluate
val_predictions = best_model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = root_mean_squared_error(y_val, val_predictions)

print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")

# Predict on the test set
test_predictions = best_model.predict(test.drop('id', axis=1))

# Save the results
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Tahminler 'submission.csv' dosyasına kaydedildi.")