import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import uniform, randint

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
    for df in [X, test]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df = df.drop(col, axis=1)

# Update categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Data preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Define base models
xgb_model = XGBRegressor(n_jobs=-1, random_state=42)
lgbm_model = LGBMRegressor(n_jobs=-1, random_state=42)
catboost_model = CatBoostRegressor(thread_count=-1, random_state=42, verbose=0)

# Create a voting ensemble
ensemble_model = VotingRegressor([
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('catboost', catboost_model)
])

# Create the pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ensemble_model)
])

# Parameters for RandomizedSearchCV
param_distributions = {
    'regressor__xgb__n_estimators': randint(100, 1000),
    'regressor__xgb__learning_rate': uniform(0.01, 0.3),
    'regressor__xgb__max_depth': randint(3, 10),
    'regressor__lgbm__n_estimators': randint(100, 1000),
    'regressor__lgbm__learning_rate': uniform(0.01, 0.3),
    'regressor__lgbm__max_depth': randint(3, 10),
    'regressor__catboost__iterations': randint(100, 1000),
    'regressor__catboost__learning_rate': uniform(0.01, 0.3),
    'regressor__catboost__depth': randint(3, 10)
}

# Hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, n_jobs=-1, random_state=42)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform hyperparameter tuning and fit the model
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Predict on the validation set and evaluate
val_predictions = best_model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
mae = mean_absolute_error(y_val, val_predictions)

print(f"Validation MSE: {mse}")
print(f"Validation MAE: {mae}")

# Predict on the test set
test_predictions = best_model.predict(test.drop('id', axis=1))

# Save the results
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Tahminler 'submission.csv' dosyasına kaydedildi.")