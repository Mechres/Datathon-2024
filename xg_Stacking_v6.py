import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
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
import time
from tqdm import tqdm


# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test_x.csv')

# Preprocessing: Target variable
y = train['Degerlendirme Puani']
# Imputing missing target values with median (more robust to outliers)
y = y.fillna(y.median())
# Handling infinite values
y = y.replace([np.inf, -np.inf], np.nan)
y = y.fillna(y.median())
X = train.drop(['Degerlendirme Puani', 'id'], axis=1)
X = X.loc[y.index]  # Align X with y after dropping rows

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
    for col in categorical_columns:
        df[col] = df[col].astype(str)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Median imputation for robustness
    ('scaler', RobustScaler())  # RobustScaler for handling outliers
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


# Quick run and advanced stacking/blending options
quick_run = input("Do you want to do a quick run with minimal hyperparameter tuning? (yes/no): ").lower() == 'yes'
use_complex_meta_learner = input("Do you want to use complex meta-learners (Gradient Boosting, Random Forest)? (yes/no): ").lower() == 'yes'
use_blending = input("Do you want to use blending? (yes/no): ").lower() == 'yes'


# Models dictionary - structure improved for easier management
models = {
    "1": {
        "name": "XGBoost",
        "regressor": XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1),
        "param_distributions": {
            'regressor__n_estimators': [100, 200] if quick_run else [50, 100, 200],
            'regressor__max_depth': [3, 5] if quick_run else [3, 5, 7],
            'regressor__learning_rate': [0.1, 0.01] if quick_run else [0.1, 0.01, 0.001],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0],
        }
    },
    "2": {
        "name": "Random Forest",
        "regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
        "param_distributions": {
            'regressor__n_estimators': [100, 200] if quick_run else [50, 100, 200],
            'regressor__max_depth': [5, 10] if quick_run else [None, 5, 10],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2],
            'regressor__max_features': ['sqrt', 'log2']
        }
    },
    "3": {
        "name": "Gradient Boosting",
        "regressor": GradientBoostingRegressor(random_state=42),
        "param_distributions": {
            'regressor__n_estimators': [100, 200] if quick_run else [50, 100, 200],
            'regressor__max_depth': [3, 5] if quick_run else [3, 5, 7],
            'regressor__learning_rate': [0.1, 0.01] if quick_run else [0.1, 0.01, 0.001],
            'regressor__subsample': [0.8, 1.0]
        }
    },
    "4": {
        "name": "SVR",
        "regressor": SVR(),
        "param_distributions": {
            'regressor__C': [1, 10] if quick_run else [0.1, 1, 10],
            'regressor__kernel': ['rbf', 'linear'] if quick_run else ['rbf', 'linear', 'poly'],
            'regressor__gamma': ['scale', 'auto']
        }
    },
    "5": {
        "name": "CatBoost",
        "regressor": CatBoostRegressor(random_state=42, verbose=0, loss_function='RMSE'),  # Specify RMSE loss directly
        "param_distributions": {
            'regressor__iterations': [100, 200] if quick_run else [50, 100, 200],
            'regressor__depth': [4, 6] if quick_run else [4, 6, 8],
            'regressor__learning_rate': [0.1, 0.01] if quick_run else [0.1, 0.01, 0.001],
            # Additional CatBoost-specific hyperparameters:
            'regressor__l2_leaf_reg': [1, 3, 5],  # Regularization
            'regressor__border_count': [32, 64, 128] if quick_run else [32, 64, 128, 254],  # Binning precision
        }
    },
    "6": {
        "name": "Extra Trees",
        "regressor": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "param_distributions": {
            'regressor__n_estimators': [100, 200] if quick_run else [50, 100, 200],
            'regressor__max_depth': [5, 10] if quick_run else [None, 5, 10],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2],
            'regressor__max_features': ['sqrt', 'log2']
        }
    }
}

# User model selection (improved for clarity)
print("Available Models:")
for key, value in models.items():
    print(f"{key}: {value['name']}")

selected_models = input("Please enter the numbers of the models you want to use, separated by commas (or type 'all'): ")

if selected_models.lower() == "all":
    selected_models = list(models.keys())
else:
    selected_models = selected_models.split(",")


# Function to create base models (improved with error handling)
def create_base_models(models_dict, selected_models):
    base_models = []
    for model_num in selected_models:
        if model_num in models_dict:
            model_data = models_dict[model_num]
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model_data['regressor'])
            ])
            base_models.append((model_data['name'], model))
        else:
            print(f"Invalid model number: {model_num}, skipping.")
    return base_models

# Create base models
base_models = create_base_models(models, selected_models)

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
        verbose=1
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

# Train and evaluate base models (using tqdm for progress bar)
tuned_base_models = []
for model_name, model in tqdm(base_models, desc="Training models"):
    print(f"\nModel: {model_name}")
    param_dist = next((m['param_distributions'] for m in models.values() if m['name'] == model_name), {})
    tuned_model = train_and_evaluate(model, param_dist, X_train, y_train, X_val, y_val)
    tuned_base_models.append((model_name, tuned_model))

# Stacking/Blending
if use_blending:
    print("\nPerforming Blending...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    blend_train = np.zeros((X_train.shape[0], len(tuned_base_models)))
    blend_test = np.zeros((test.shape[0], len(tuned_base_models)))

    for i, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Fold {i+1}")
        X_blend_train, X_blend_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_blend_train, y_blend_val = y_train[train_index], y_train[val_index]

        for j, (model_name, model) in enumerate(tuned_base_models):
            model.fit(X_blend_train, y_blend_train)
            blend_train[val_index, j] = model.predict(X_blend_val)
            blend_test[:, j] += model.predict(test.drop('id', axis=1)) / kf.n_splits

    # Train meta-learner on blended predictions
    meta_learner = Ridge()  # Default to Ridge if not using complex meta-learners
    if use_complex_meta_learner:
        meta_learners = {
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1)
        }
        for meta_learner_name, meta_learner in meta_learners.items():
            meta_learner.fit(blend_train, y_train)
            final_predictions = meta_learner.predict(blend_test)
            print(f"\nBlending Results with {meta_learner_name} Meta-Learner:")
            # Unscale predictions and true values for meaningful RMSE
            final_predictions_unscaled = target_scaler.inverse_transform(final_predictions.reshape(-1, 1)).ravel()

            # Save results
            submission = pd.DataFrame({
                'id': test['id'],
                'Degerlendirme Puani': final_predictions_unscaled
            })
            submission.to_csv(f'submission_blending_{meta_learner_name.lower().replace(" ", "_")}.csv', index=False)

            print(f"Blending predictions with {meta_learner_name} saved.")

    else:
        meta_learner.fit(blend_train, y_train)
        final_predictions = meta_learner.predict(blend_test)
        print("\nBlending Results:")
        # Unscale predictions and true values for meaningful RMSE
        final_predictions_unscaled = target_scaler.inverse_transform(final_predictions.reshape(-1, 1)).ravel()

        # Save results
        submission = pd.DataFrame({
            'id': test['id'],
            'Degerlendirme Puani': final_predictions_unscaled
        })
        submission.to_csv('submission_blending_ridge.csv', index=False)

        print("Blending predictions saved.")

else:  # Regular Stacking
    print("\nPerforming Stacking...")
    meta_learner = Ridge()  # Default to Ridge if not using complex meta-learners
    if use_complex_meta_learner:
        meta_learners = {
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1)
        }
        for meta_learner_name, meta_learner in meta_learners.items():
            stacking_regressor = StackingRegressor(
                estimators=tuned_base_models,
                final_estimator=meta_learner,
                cv=3 if quick_run else 5
            )

            # Train the stacking regressor
            print(f"\nTraining Stacking Regressor with {meta_learner_name}...")
            start_time = time.time()
            stacking_regressor.fit(X_train, y_train)
            end_time = time.time()
            print(f"Stacking Regressor Training time: {end_time - start_time:.2f} seconds")

            # Performance on validation set (scaled and unscaled)
            val_predictions = stacking_regressor.predict(X_val)

            # Unscale predictions and true values for meaningful RMSE
            val_predictions_unscaled = target_scaler.inverse_transform(val_predictions.reshape(-1, 1)).ravel()
            y_val_unscaled = target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()

            mse = mean_squared_error(y_val_unscaled, val_predictions_unscaled)
            rmse = np.sqrt(mse)
            print(f"Stacking Regressor Validation RMSE (unscaled): {rmse}")

            # Cross-validation for more robust evaluation (scaled and unscaled)
            print("Performing cross-validation...")
            cv_scores = cross_val_score(stacking_regressor, X, y_scaled, cv=3 if quick_run else 5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            print(f"Cross-validation RMSE (scaled): {cv_rmse.mean()} (+/- {cv_rmse.std() * 2})")

            # Unscale cross-validation scores for a more interpretable metric
            cv_rmse_unscaled = target_scaler.inverse_transform(cv_rmse.reshape(-1, 1)).ravel()
            print(f"Cross-validation RMSE (unscaled): {cv_rmse_unscaled.mean()} (+/- {cv_rmse_unscaled.std() * 2})")

            # Predictions on test set
            print("Generating predictions on test set...")
            test_predictions = stacking_regressor.predict(test.drop('id', axis=1))

            # Unscale test predictions to original scale
            test_predictions_original_scale = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).ravel()

            # Save results
            submission = pd.DataFrame({
                'id': test['id'],
                'Degerlendirme Puani': test_predictions_original_scale
            })
            submission.to_csv(f'submission_stacking_{meta_learner_name.lower().replace(" ", "_")}.csv', index=False)

            print(f"Stacking Regressor predictions with {meta_learner_name} saved.")

    else:
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

        # Performance on validation set (scaled and unscaled)
        val_predictions = stacking_regressor.predict(X_val)

        # Unscale predictions and true values for meaningful RMSE
        val_predictions_unscaled = target_scaler.inverse_transform(val_predictions.reshape(-1, 1)).ravel()
        y_val_unscaled = target_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()

        mse = mean_squared_error(y_val_unscaled, val_predictions_unscaled)
        rmse = np.sqrt(mse)
        print(f"Stacking Regressor Validation RMSE (unscaled): {rmse}")

        # Cross-validation for more robust evaluation (scaled and unscaled)
        print("Performing cross-validation...")
        cv_scores = cross_val_score(stacking_regressor, X, y_scaled, cv=3 if quick_run else 5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        print(f"Cross-validation RMSE (scaled): {cv_rmse.mean()} (+/- {cv_rmse.std() * 2})")

        # Unscale cross-validation scores for a more interpretable metric
        cv_rmse_unscaled = target_scaler.inverse_transform(cv_rmse.reshape(-1, 1)).ravel()
        print(f"Cross-validation RMSE (unscaled): {cv_rmse_unscaled.mean()} (+/- {cv_rmse_unscaled.std() * 2})")

        # Predictions on test set
        print("Generating predictions on test set...")
        test_predictions = stacking_regressor.predict(test.drop('id', axis=1))

        # Unscale test predictions to original scale
        test_predictions_original_scale = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).ravel()

        # Save results
        submission = pd.DataFrame({
            'id': test['id'],
            'Degerlendirme Puani': test_predictions_original_scale
        })
        submission.to_csv('submission_stacking_ridge.csv', index=False)

        print("Stacking Regressor predictions saved.")