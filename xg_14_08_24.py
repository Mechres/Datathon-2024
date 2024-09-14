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
from sklearn.feature_selection import SelectKBest, f_regression
import time  # For timing execution

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# Reduce dataset size for faster experimentation
sample_fraction = 0.3  # Adjust as needed
train = train.sample(frac=sample_fraction, random_state=42)

# Preprocessing: Target variable
y = train['Degerlendirme Puani']
y = y.fillna(y.mean())
y = y.replace([np.inf, -np.inf], np.nan).dropna()
X = train.drop(['Degerlendirme Puani', 'id'], axis=1)
X = X.loc[y.index]  # Align X and y after dropping NaN values

# Preprocessing: Date columns
date_columns = ['Dogum Tarihi']
for col in date_columns:
    for df in [X, test]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_weekday'] = df[col].dt.weekday
        df.drop(col, axis=1, inplace=True)

# Preprocessing: Categorical & Numeric columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

for df in [X, test]:
    df[categorical_columns] = df[categorical_columns].astype(str)

# Preprocessing pipeline
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

# Feature selection (optional, uncomment if needed)
feature_selector = SelectKBest(f_regression, k=50)  # Adjust 'k' as needed

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "1": ("XGBoost", XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1)),
    "2": ("Random Forest", RandomForestRegressor(random_state=42, n_jobs=-1)),
    "3": ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
    "4": ("SVR", SVR())
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

# Loop for selected models
for model_num in selected_models:
    if model_num not in models:
        print(f"Invalid model number: {model_num}, skipping.")
        continue

    model_name, regressor = models[model_num]

    # Pipeline with optional feature selection
    model = Pipeline([
        ('preprocessor', preprocessor),
        # ('feature_selector', feature_selector),  # Uncomment if using feature selection
        ('regressor', regressor)
    ])

    # Parameter grids
    param_distributions = {
        'regressor__n_estimators': [100, 200, 300],  # Expanded range
        'regressor__max_depth': [3, 5, 7],  # Expanded range
    }

    if model_name == "XGBoost":
        param_distributions['regressor__learning_rate'] = [0.01, 0.1, 0.2]  # Expanded range
    elif model_name == "Gradient Boosting":
        param_distributions['regressor__learning_rate'] = [0.01, 0.1, 0.2]
    elif model_name == "SVR":
        param_distributions = {
            'regressor__C': [0.1, 1, 10],
            'regressor__kernel': ['rbf', 'linear', 'poly'],  # Expanded kernels
            'regressor__gamma': ['scale', 'auto'] + [0.1, 1, 10]  # Expanded gamma
        }

    # RandomizedSearchCV
    n_iter = 10  # Increased iterations for better exploration
    cv = 5  # Increased folds for more robust evaluation
    start_time = time.time()

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2  # Show progress
    )

    # Training and evaluation
    print(f"\nModel: {model_name}")
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

    # Predictions on test set
    test_predictions = random_search.predict(test.drop('id', axis=1))

    # Save results
    submission = pd.DataFrame({
        'id': test['id'],
        'Degerlendirme Puani': test_predictions
    })
    submission.to_csv(f'submission_{model_name}.csv', index=False)

    print(f"{model_name} predictions saved.")