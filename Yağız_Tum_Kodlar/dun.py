import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    columns_to_keep = [
        'Degerlendirme Puani', 'Universite Turu', 'Burs Aliyor mu?', 'Cinsiyet',
        'Daha Once Baska Bir Universiteden Mezun Olmus', 'Universite Adi',
        'Lise Turu', 'Universite Not Ortalamasi', 'Universite Kacinci Sinif',
        'Anne Egitim Durumu', 'Anne Calisma Durumu', 'Anne Sektor',
        'Baba Egitim Durumu', 'Baba Calisma Durumu', 'Baba Sektor',
        'Lise Mezuniyet Notu', 'Baska Bir Kurumdan Burs Aliyor mu?',
        'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
        'Profesyonel Bir Spor Daliyla Mesgul musunuz?',
        'Spor Dalindaki Rolunuz Nedir?', 'Aktif olarak bir STK üyesi misiniz?',
        'Stk Projesine Katildiniz Mi?', 'Girisimcilikle Ilgili Deneyiminiz Var Mi?',
        'Ingilizce Biliyor musunuz?', 'Ingilizce Seviyeniz?', 'id'
    ]

    df = df.loc[:, columns_to_keep]

    # Apply mappings (keep existing mappings)
    lise_turu_mapping = {'İmam Hatip Lisesi': 0, 'Diğer': 0, 'Devlet': 0, 'Düz lise': 0,
                         'Düz Lise': 0, 'Meslek lisesi': 1, 'Meslek': 1, 'Meslek Lisesi': 1, 'Özel': 1,
                         "Özel Lisesi": 1,
                         "Özel lisesi": 1, "Özel Lise": 1, 'Anadolu Lisesi': 2, 'Anadolu lisesi': 2, 'Fen lisesi': 3,
                         'Fen Lisesi': 3}

    anne_egitim_mapping = {
        'İlkokul': 0, 'İlkokul Mezunu': 0, 'İLKOKUL MEZUNU': 0, 'Eğitimi yok': 0, 'EĞİTİM YOK': 0, 'Eğitim Yok': 0,
        'Ortaokul': 1, 'Ortaokul Mezunu': 1, 'ORTAOKUL MEZUNU': 1,
        'Lise': 2, 'LİSE': 2, 'Lise Mezunu': 2,
        'Üniversite': 3, 'ÜNİVERSİTE': 3, 'Üniversite Mezunu': 3,
        'Yüksek Lisans': 4, 'YÜKSEK LİSANS': 4, 'Yüksek Lisans / Doktora': 4, 'Yüksek Lisans / Doktara': 4,
        'Doktora': 5, 'DOKTORA': 5
    }

    # Apply other mappings similarly...

    df['Lise Turu'] = df['Lise Turu'].map(lise_turu_mapping)
    df['Anne Egitim Durumu'] = df['Anne Egitim Durumu'].map(anne_egitim_mapping)
    # Apply other mappings...

    # Data cleaning and transformations
    df['Lise Mezuniyet Notu'] = df['Lise Mezuniyet Notu'].replace({'Not ortalaması yok': '0'})
    df['Lise Mezuniyet Notu'] = df['Lise Mezuniyet Notu'].apply(convert_to_range)
    df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].apply(convert_to_range)

    # Handle missing values
    df = handle_missing_values(df)

    # Feature Engineering
    df = create_features(df)
    # Check for NaN values after preprocessing
    check_nan_values(df)

    # If there are still NaN values, drop those rows
    df = df.dropna()

    # Convert to lowercase and encode categorical variables
    df = preprocess_categorical(df)

    return df


def convert_to_range(value):
    value = str(value).strip().lower()

    if '0 - 24' in value or '0 - 25' in value or '25 - 49' in value or '44-0' in value or '0 - 25' in value or '25 - 50' in value or '54-45' in value or '25 - 50' in value:
        return '0-50'
    elif '50 - 74' in value or '50 - 75' in value or '54-45' in value or '69-55' in value or '84-70' in value or '3.00-2.50' in value or '84-70' in value:
        return '50-75'
    elif '75 - 100' in value or '100-85' in value:
        return '75-100'
    elif '3.00 - 4.00' in value or '3.50-3.00' in value or '3.50-3' in value or '4.00-3.50' in value:
        return '75-100'
    elif '2.50 ve altı' in value:
        return '0-50'
    else:
        return '0'


def handle_missing_values(df):
    # Kategorik değişkenler için mod (en sık görülen değer) ile doldurma
    categorical_columns = [
        "Universite Turu", "Daha Once Baska Bir Universiteden Mezun Olmus",
        "Baska Bir Kurumdan Burs Aliyor mu?",
        "Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?",
        "Profesyonel Bir Spor Daliyla Mesgul musunuz?",
        "Spor Dalindaki Rolunuz Nedir?", "Aktif olarak bir STK üyesi misiniz?",
        "Stk Projesine Katildiniz Mi?", "Girisimcilikle Ilgili Deneyiminiz Var Mi?",
        "Ingilizce Biliyor musunuz?", "Ingilizce Seviyeniz?", "Cinsiyet"
    ]

    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Sayısal değişkenler için medyan ile doldurma
    numeric_columns = [
        "Universite Not Ortalamasi", "Lise Mezuniyet Notu",
        "Anne Egitim Durumu", "Baba Egitim Durumu",
        "Anne Calisma Durumu", "Baba Calisma Durumu",
        "Anne Sektor", "Baba Sektor"
    ]

    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])



    # Özel durumlar
    df.loc[df["Universite Turu"] == "nan", "Universite Turu"] = "devlet"
    df.loc[df[
               "Daha Once Baska Bir Universiteden Mezun Olmus"] == "nan", "Daha Once Baska Bir Universiteden Mezun Olmus"] = "hayır"
    df.loc[df["Baska Bir Kurumdan Burs Aliyor mu?"] == "nan", "Baska Bir Kurumdan Burs Aliyor mu?"] = "hayır"
    df.loc[df[
               "Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] == "nan", "Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = "hayır"
    df.loc[df[
               "Profesyonel Bir Spor Daliyla Mesgul musunuz?"] == "nan", "Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = "hayır"
    df.loc[df["Spor Dalindaki Rolunuz Nedir?"] == "nan", "Spor Dalindaki Rolunuz Nedir?"] = "diğer"
    df.loc[df["Aktif olarak bir STK üyesi misiniz?"] == "nan", "Aktif olarak bir STK üyesi misiniz?"] = "hayır"
    df.loc[df["Stk Projesine Katildiniz Mi?"] == "nan", "Stk Projesine Katildiniz Mi?"] = "hayır"
    df.loc[
        df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] == "nan", "Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = "hayır"
    df.loc[df["Ingilizce Biliyor musunuz?"] == "nan", "Ingilizce Biliyor musunuz?"] = "hayır"
    df.loc[df["Ingilizce Seviyeniz?"] == "nan", "Ingilizce Seviyeniz?"] = "başlangıç"
    df.loc[df["Cinsiyet"] == "nan", "Cinsiyet"] = "erkek"
    df.loc[df["Cinsiyet"] == "belirtmek istemiyorum", "Cinsiyet"] = "kadın"

    # Kalan NaN değerlerini kontrol et ve raporla
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print("Remaining NaN values:")
        print(nan_counts[nan_counts > 0])

    return df

def create_features(df):
    # Önce sütunları sayısal değerlere dönüştürün
    df['Anne Egitim Durumu'] = pd.to_numeric(df['Anne Egitim Durumu'], errors='coerce')
    df['Baba Egitim Durumu'] = pd.to_numeric(df['Baba Egitim Durumu'], errors='coerce')
    df['Anne Calisma Durumu'] = pd.to_numeric(df['Anne Calisma Durumu'], errors='coerce')
    df['Baba Calisma Durumu'] = pd.to_numeric(df['Baba Calisma Durumu'], errors='coerce')

    df['Aile Egitim Seviyesi'] = df['Anne Egitim Durumu'] + df['Baba Egitim Durumu']
    df['Aile Calisma Durumu'] = df['Anne Calisma Durumu'] + df['Baba Calisma Durumu']
    df['Aile Sosyoekonomik Durum'] = df['Aile Egitim Seviyesi'] + df['Aile Calisma Durumu']

    activity_scores = {'evet': 1, 'hayır': 0}
    activity_columns = ['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
                        'Profesyonel Bir Spor Daliyla Mesgul musunuz?',
                        'Aktif olarak bir STK üyesi misiniz?',
                        'Girisimcilikle Ilgili Deneyiminiz Var Mi?']

    df['Aktivite Skoru'] = df[activity_columns].applymap(lambda x: activity_scores.get(str(x).lower(), 0)).sum(axis=1)

    return df


def preprocess_categorical(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].applymap(lambda x: str(x).lower())

    return pd.get_dummies(df, columns=categorical_columns, drop_first=False)


# Model Training and Evaluation
def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = {}

    for name, model in models.items():
        if name == 'Linear Regression':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            param_grid = get_param_grid(name)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
                                       scoring='neg_root_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            y_pred = model.predict(X_test)
            print(f'Best parameters for {name}:', grid_search.best_params_)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {'model': model, 'rmse': rmse}
        print(f'{name} RMSE: {rmse}')

        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            print(f'{name} Feature Importances:\n', feature_importances.head(10))

    return results


def get_param_grid(model_name):
    if model_name == 'Random Forest':
        return {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'XGBoost':
        return {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
def check_data_types(df):
    print(df[['Anne Egitim Durumu', 'Baba Egitim Durumu', 'Anne Calisma Durumu', 'Baba Calisma Durumu']].dtypes)
    print(df[['Anne Egitim Durumu', 'Baba Egitim Durumu', 'Anne Calisma Durumu', 'Baba Calisma Durumu']].head())

def check_nan_values(df):
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print("Columns with NaN values:")
        print(nan_counts[nan_counts > 0])
    else:
        print("No NaN values found in the dataset.")


# Main execution
if __name__ == "__main__":
    df = load_and_preprocess_data("train.csv")
    X = df.drop(columns=['id', 'Degerlendirme Puani'])
    y = df['Degerlendirme Puani']

    # Final check for NaN values before model training
    check_nan_values(X)
    check_nan_values(pd.DataFrame(y))

    results = train_and_evaluate_models(X, y)

    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f'\nBest Model: {best_model[0]} with RMSE: {best_model[1]["rmse"]}')