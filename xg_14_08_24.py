import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Load datasets (consider using a smaller subset for faster experimentation)
train = pd.read_csv('train.csv').sample(frac=0.3, random_state=42)  # Use 30% of data
test = pd.read_csv('test_x.csv')

# Data preprocessing (same as before)
y = train['Degerlendirme Puani']
y = y.fillna(y.mean())
y = y.replace([np.inf, -np.inf], np.nan).dropna()

X = train.drop(['Degerlendirme Puani', 'id'], axis=1)
X = X.loc[y.index]

# Transform date columns (same as before)
date_columns = ['Dogum Tarihi']
for col in date_columns:
    X[col] = pd.to_datetime(X[col], errors='coerce')
    X[f'{col}_year'] = X[col].dt.year
    X[f'{col}_month'] = X[col].dt.month
    X[f'{col}_weekday'] = X[col].dt.weekday
    X = X.drop(col, axis=1)

    test[col] = pd.to_datetime(test[col], errors='coerce')
    test[f'{col}_year'] = test[col].dt.year
    test[f'{col}_month'] = test[col].dt.month
    test[f'{col}_weekday'] = test[col].dt.weekday
    test = test.drop(col, axis=1)

categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

X[categorical_columns] = X[categorical_columns].astype(str)
test[categorical_columns] = test[categorical_columns].astype(str)

# Create data preprocessing pipeline (same as before)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
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

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Available models with early stopping where applicable
models = {
    "1": ("XGBoost", XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1, early_stopping_rounds=10)),
    "2": ("Random Forest", RandomForestRegressor(random_state=42, n_jobs=-1)),
    "3": ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    "4": ("SVR", SVR())
}

# User model selection (same as before)
print("Available Models:")
for key, value in models.items():
    print(f"{key}: {value[0]}")

selected_models = input("Please enter the numbers of the models you want to use, separated by commas (or type 'all'): ")

if selected_models.lower() == "all":
    selected_models = list(models.keys())
else:
    selected_models = selected_models.split(",")

# Loop for selected models
for model_num in selected_models:
    if model_num not in models:
        print(f"Invalid model number: {model_num}, skipping.")
        continue

    model_name, regressor = models[model_num]

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    # Simplified parameter grids
    if model_name == "XGBoost":
        param_distributions = {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 5]
        }
    elif model_name == "Random Forest":
        param_distributions = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [5, 10]
        }
    elif model_name == "Gradient Boosting":
        param_distributions = {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1]
        }
    elif model_name == "SVR":
        param_distributions = {
            'regressor__C': [0.1, 1, 10],
            'regressor__kernel': ['rbf', 'linear']
        }

    # RandomizedSearchCV settings
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=5,  # Reduced number of iterations
        cv=3,  # Reduced number of folds
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Training and evaluation
    print(f"\nModel: {model_name}")
    print("Training started...")
    random_search.fit(X_train, y_train)
    print("Training completed.")

    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation score:", -random_search.best_score_)

    # Make predictions using the best model
    best_model = random_search.best_estimator_

    # Performance evaluation on validation set
    val_predictions = best_model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    rmse = np.sqrt(mse)

    print(f"Validation MSE: {mse}")
    print(f"Validation RMSE: {rmse}")

    # Make predictions on test set
    test_predictions = best_model.predict(test.drop('id', axis=1))

    # Save results
    submission = pd.DataFrame({
        'id': test['id'],
        'Degerlendirme Puani': test_predictions
    })
    submission.to_csv(f'submission_{model_name}.csv', index=False)

    print(f"{model_name} predictions saved to 'submission_{model_name}.csv'.")