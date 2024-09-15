import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('train.csv')

print(train_data.columns)

X_train = train_data.drop('Degerlendirme Puani', axis=1)
y_train = train_data['Degerlendirme Puani']

test_data = pd.read_csv('test_x.csv')

numerik_degerler = ['Universite Not Ortalamasi', 'Lise Mezuniyet Notu', 'Kardes Sayisi']
kategorik_degerler = ['Cinsiyet', 'Universite Turu', 'Burs Aliyor mu?', 'Lise Turu']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerik_degerler),
        ('cat', OneHotEncoder(handle_unknown='ignore'), kategorik_degerler)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(mean_squared_error))

# Calculate and print the mean and standard deviation of MSE
mean_mse = np.mean(cv_scores)
std_mse = np.std(cv_scores)
print(f"Cross-validation Mean Squared Error: {mean_mse:.4f} (+/- {std_mse:.4f})")

# Calculate and print the Root Mean Squared Error (RMSE)
mean_rmse = np.sqrt(mean_mse)
std_rmse = np.sqrt(std_mse)
print(f"Cross-validation Root Mean Squared Error: {mean_rmse:.4f} (+/- {std_rmse:.4f})")



predictions = model.predict(test_data)


test_data['Predicted_Degerlendirme_Puani'] = predictions