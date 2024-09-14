import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import time
from tqdm import tqdm


# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

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
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df.drop(col, axis=1, inplace=True)

# Preprocessing: Categorical & Numeric columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

for df in [X, test]:
    df[categorical_columns] = df[categorical_columns].astype(str)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Models
models = {
    "1": ("XGBoost", XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1)),
    "2": ("Random Forest", RandomForestRegressor(random_state=42, n_jobs=-1)),
    "3": ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    "4": ("SVR", SVR()),
    "5": ("CatBoost", CatBoostRegressor(random_state=42, verbose=0)),
    "6": ("LightGBM", LGBMRegressor(random_state=42, n_jobs=-1)),
    "7": ("Extra Trees", ExtraTreesRegressor(random_state=42, n_jobs=-1))
}


# User model selection
print("Available Models:")
for key, value in models.items():
    print(f"{key}: {value[0]}")

selected_models = input("Please enter the numbers of the models you want to use, separated by commas (or type 'all'): ")

if selected_models.lower() == "all":
    selected_models = list(models.keys())
else:
    selected_models = selected_models.split(",")

# Quick run option
quick_run = input("Do you want to do a quick run with minimal hyperparameter tuning? (yes/no): ").lower() == 'yes'

# Function to create base models
def create_base_models(models_dict, selected_models):
    base_models = []
    for model_num in selected_models:
        if model_num in models_dict:
            model_name, regressor = models_dict[model_num]
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', regressor)
            ])
            base_models.append((model_name, model))
        else:
            print(f"Invalid model number: {model_num}, skipping.")
    return base_models

# Create base models
base_models = create_base_models(models, selected_models)

# Hyperparameter grids
param_distributions = {
    "XGBoost": {
        'regressor__n_estimators': [100, 200] if quick_run else [100, 200, 300],
        'regressor__max_depth': [3, 5] if quick_run else [3, 5, 7],
        'regressor__learning_rate': [0.1, 0.2] if quick_run else [0.01, 0.1, 0.2],
    },
    "Random Forest": {
        'regressor__n_estimators': [100, 200] if quick_run else [100, 200, 300],
        'regressor__max_depth': [5, 10] if quick_run else [5, 10, None],
    },
    "Gradient Boosting": {
        'regressor__n_estimators': [100, 200] if quick_run else [100, 200, 300],
        'regressor__max_depth': [3, 5] if quick_run else [3, 5, 7],
        'regressor__learning_rate': [0.1, 0.2] if quick_run else [0.01, 0.1, 0.2],
    },
    "SVR": {
        'regressor__C': [1, 10] if quick_run else [0.1, 1, 10],
        'regressor__kernel': ['rbf', 'linear'] if quick_run else ['rbf', 'linear', 'poly'],
    },
    "CatBoost": {
        'regressor__iterations': [100, 200] if quick_run else [100, 200, 300],
        'regressor__depth': [4, 6] if quick_run else [4, 6, 8],
        'regressor__learning_rate': [0.1, 0.2] if quick_run else [0.01, 0.1, 0.2],
    },
    "LightGBM": {
        'regressor__n_estimators': [100, 200] if quick_run else [100, 200, 300],
        'regressor__max_depth': [3, 5] if quick_run else [3, 5, 7],
        'regressor__learning_rate': [0.1, 0.2] if quick_run else [0.01, 0.1, 0.2],
    },
    "Extra Trees": {
        'regressor__n_estimators': [100, 200] if quick_run else [100, 200, 300],
        'regressor__max_depth': [5, 10] if quick_run else [5, 10, None],
    }
}
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Function for model training and evaluation
def train_and_evaluate(model, param_dist, X_train, y_train, X_val, y_val):
    n_iter = 5 if quick_run else 10
    cv = 3 if quick_run else 5
    start_time = time.time()

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    random_search.fit(X_train, y_train)
    end_time = time.time()

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score (RMSE): {np.sqrt(-random_search.best_score_)}")
    print(f"Training time: {end_time - start_time:.2f} seconds")

    val_predictions = random_search.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    rmse = np.sqrt(mse)
    print(f"Validation RMSE: {rmse}")

    return random_search.best_estimator_

# Train and evaluate base models
tuned_base_models = []
for model_name, model in tqdm(base_models, desc="Training models"):
    print(f"\nModel: {model_name}")
    tuned_model = train_and_evaluate(model, param_distributions[model_name], X_train, y_train, X_val, y_val)
    tuned_base_models.append((model_name, tuned_model))

# Stacking Regressor
meta_learner = Ridge()
stacking_regressor = StackingRegressor(
    estimators=tuned_base_models,
    final_estimator=meta_learner,
    cv=3 if quick_run else 5
)

# Train the stacking regressor
print("\nTraining Stacking Regressor...")
start_time = time.time()
stacking_regressor.fit(X_train, y_train)
end_time = time.time()
print(f"Stacking Regressor Training time: {end_time - start_time:.2f} seconds")

# Performance on validation set
val_predictions = stacking_regressor.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
print(f"Stacking Regressor Validation RMSE: {rmse}")

# Cross-validation for more robust evaluation
print("Performing cross-validation...")
cv_scores = cross_val_score(stacking_regressor, X, y_scaled, cv=3 if quick_run else 5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-validation RMSE: {cv_rmse.mean()} (+/- {cv_rmse.std() * 2})")

# Predictions on test set
print("Generating predictions on test set...")
test_predictions = stacking_regressor.predict(test.drop('id', axis=1))
test_predictions_original_scale = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).ravel()

# Save results
submission = pd.DataFrame({
    'id': test['id'],
    'Degerlendirme Puani': test_predictions_original_scale
})
submission.to_csv('submission_improved_stacking.csv', index=False)

print("Improved Stacking Regressor predictions saved.")