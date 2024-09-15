import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
from sklearn.impute import SimpleImputer
import category_encoders as ce

# Load datasets
train = pd.read_csv('train.csv', low_memory=False)
test = pd.read_csv('test_x.csv', low_memory=False)

# Preprocessing: Target variable
y = train['Degerlendirme Puani']
y = y.fillna(y.mean())
y = y.replace([np.inf, -np.inf], np.nan).dropna()
X = train.drop(['Degerlendirme Puani', 'id'], axis=1)
X = X.loc[y.index]

# Feature scaling for target variable
target_scaler = RobustScaler()
y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# Preprocessing: Date columns
date_columns = ['Dogum Tarihi']
for col in date_columns:
    for df in [X, test]:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        df[f'{col}_quarter'] = df[col].dt.quarter
        df.drop(col, axis=1, inplace=True)


# Feature Engineering
def add_features(df):
    # Age calculation
    current_year = 2024  # Adjust if necessary
    df['Age'] = current_year - df['Dogum Tarihi_year']

    # Interactions between numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            df[f'{numeric_cols[i]}_{numeric_cols[j]}_interaction'] = df[numeric_cols[i]] * df[numeric_cols[j]]

    return df


X = add_features(X)
test = add_features(test)

# Ensure X and test have the same columns
common_columns = X.columns.intersection(test.columns)
X = X[common_columns]
test = test[common_columns]

# Identify categorical and numeric columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Imputation
imputer = SimpleImputer(strategy='constant', fill_value='missing')
X[categorical_columns] = imputer.fit_transform(X[categorical_columns])
test[categorical_columns] = imputer.transform(test[categorical_columns])

# Target Encoding for categorical variables
encoder = ce.TargetEncoder()
X_encoded = encoder.fit_transform(X[categorical_columns], y_scaled)
test_encoded = encoder.transform(test[categorical_columns])

# Combine encoded categorical features with numeric features
X_final = pd.concat([X_encoded, X[numeric_columns]], axis=1)
test_final = pd.concat([test_encoded, test[numeric_columns]], axis=1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_final, y_scaled, test_size=0.2, random_state=42)

# Create CatBoost pools
train_pool = Pool(X_train, y_train, cat_features=list(X_encoded.columns))
val_pool = Pool(X_val, y_val, cat_features=list(X_encoded.columns))

# Hyperparameter tuning
param_dist = {
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.03, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 1, 10],
    'random_strength': [1, 10, 100],
    'scale_pos_weight': [1, 10, 50],
    'one_hot_max_size': [2, 10, 25],
    'min_data_in_leaf': [1, 5, 10, 15, 20],
    'max_bin': [200, 400, 600]
}

model = CatBoostRegressor(
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=50,  # Increased number of iterations
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=list(X_encoded.columns))

# Best model and parameters
best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

# Evaluate on validation set
val_predictions = best_model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
print(f"Validation RMSE (scaled): {rmse}")

# Unscale RMSE
rmse_unscaled = target_scaler.inverse_transform([[rmse]])[0][0]
print(f"Validation RMSE (unscaled): {rmse_unscaled}")

# Generate predictions on test set
test_predictions = best_model.predict(test_final)
test_predictions_original_scale = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).ravel()

# Save results
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions_original_scale
})
submission.to_csv('submission_improved_catboost.csv', index=False)

print("Improved CatBoost predictions saved.")