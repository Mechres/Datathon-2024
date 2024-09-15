import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import time

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# Remove problematic columns if they exist
columns_to_remove = ['Yas', 'Unnamed: 0']
train = train.drop(columns=columns_to_remove, errors='ignore')
test = test.drop(columns=columns_to_remove, errors='ignore')

# Reduce dataset size for faster experimentation (optional)
sample_fraction = 0.4
train = train.sample(frac=sample_fraction, random_state=42)

# Preprocessing: Target variable
y = train['Degerlendirme Puani']
y = y.fillna(y.mean())
y = y.replace([np.inf, -np.inf], np.nan).dropna()
X = train.drop(['Degerlendirme Puani'], axis=1)
X = X.loc[y.index]  # Align X and y after dropping NaN values

# Ensure consistency between train and test sets
feature_columns = X.columns.drop('id')
X_features = X[feature_columns]
test_features = test[feature_columns]

# Preprocessing: Categorical & Numeric columns
categorical_columns = X_features.select_dtypes(include=['object']).columns
numeric_columns = X_features.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Feature selection
feature_selector = SelectKBest(f_regression, k=50)  # Select top 50 features

# CatBoost model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', feature_selector),
    ('regressor', CatBoostRegressor(verbose=0, thread_count=-1))
])

# Parameter grid for CatBoost
param_distributions = {
    'regressor__iterations': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__depth': [4, 6, 8],
    'regressor__l2_leaf_reg': [1, 3, 5, 7],
    'regressor__border_count': [32, 64, 128]
}

# RandomizedSearchCV
n_iter = 20
cv = 5
start_time = time.time()

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Training and evaluation
print("\nModel: CatBoost")
random_search.fit(X_train, y_train)
end_time = time.time()

print("Best parameters:", random_search.best_params_)
print("Best cross-validation score (RMSE):", np.sqrt(-random_search.best_score_))
print(f"Training time: {end_time - start_time:.2f} seconds")

# Performance on validation set
val_predictions = random_search.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
print(f"Validation RMSE: {rmse}")

# Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=[('catboost', random_search.best_estimator_)],
    final_estimator=Ridge(random_state=42),
    cv=KFold(n_splits=5)
)

# Train the stacking regressor
start_time = time.time()
stacking_regressor.fit(X_features, y)
end_time = time.time()
print(f"\nStacking Regressor Training time: {end_time - start_time:.2f} seconds")

# Predictions on test set
test_predictions = stacking_regressor.predict(test_features)

# Save results
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions
})
submission.to_csv('submission_stacking_catboost_optimized.csv', index=False)

print("Stacking Regressor predictions saved.")