import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor
from tpot import TPOTRegressor  # Import TPOT

def tpot(X_train, X_test, y_train, y_test):
    tpot_config = {
        'xgboost.XGBRegressor': {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3]
        },
        'catboost.CatBoostRegressor': {
            'iterations': [100, 200, 500, 1000],
            'depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7],
            'random_strength': [1, 3, 5, 7],
            'bagging_temperature': [0, 0.5, 1, 1.5],
            'od_type': ['IncToDec', 'Iter'],
            'od_wait': [10, 20, 30]
        },
        'sklearn.ensemble.GradientBoostingRegressor': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'sklearn.ensemble.RandomForestRegressor': {
            'n_estimators': [100, 200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    }
    tpot_model = TPOTRegressor(
        generations=5,
        population_size=50,
        verbosity=2,
        random_state=42,
        config_dict=tpot_config,
        cv=5,
        n_jobs=-1,
        max_time_mins=120,  # Limit total runtime to 2 hours
        max_eval_time_mins=8  # Limit each pipeline evaluation to 5 minutes
    )

    tpot_model.fit(X_train, y_train)

    y_pred_tpot = tpot_model.predict(X_test)
    rmse_tpot = np.sqrt(mean_squared_error(y_test, y_pred_tpot))
    print(f'TPOT Best RMSE: {rmse_tpot}')
    print('Best pipeline:', tpot_model.fitted_pipeline_)

    return tpot_model

def yukle_ve_preprocess(dosya_yolu, is_train=True):
    df = pd.read_csv(dosya_yolu)

    if is_train:
        df['Cinsiyet'] = df['Cinsiyet'].fillna('Belirtmek istemiyorum')
        df['Universite Turu'] = df['Universite Turu'].fillna('Devlet')
        tutulacak_sutunlar = [
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
    else:
        df['Cinsiyet'] = df['Cinsiyet'].fillna('Belirtmek istemiyorum')
        df['Universite Turu'] = df['Universite Turu'].fillna('Devlet')
        tutulacak_sutunlar = [
            'Universite Turu', 'Burs Aliyor mu?', 'Cinsiyet',
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
    df = df.loc[:, tutulacak_sutunlar]

    #Mappingler
    cinsiyet_mapping = {
        'Erkek': 0,
        'Kadın': 1,
        'Belirtmek istemiyorum': 3
    }

    Universite_Turu_mapping = {
        'Devlet': 0,
        'Özel': 1,
    }
    Lise_Turu_mapping = {'İmam Hatip Lisesi': 0,
                         'Diğer': 0,
                         'Devlet': 0,
                         'Düz lise': 0,
                         'Düz Lise': 0,
                         'Meslek lisesi': 1,
                         'Meslek': 1,
                         'Meslek Lisesi': 1,
                         'Özel': 1,
                         "Özel Lisesi": 1,
                         "Özel lisesi": 1,
                         "Özel Lise": 1,
                         'Anadolu Lisesi': 2,
                         'Anadolu lisesi': 2,
                         'Fen lisesi': 3,
                         'Fen Lisesi': 3
                         }
    anne_egitim_mapping = {
        'İlkokul': 0,
        'İlkokul Mezunu': 0,
        'İLKOKUL MEZUNU': 0,
        'Eğitimi yok': 0,
        'EĞİTİM YOK': 0,
        'Eğitim Yok': 0,
        'Ortaokul': 1,
        'Ortaokul Mezunu': 1,
        'ORTAOKUL MEZUNU': 1,
        'Lise': 2,
        'LİSE': 2,
        'Lise Mezunu': 2,
        'Üniversite': 3,
        'ÜNİVERSİTE': 3,
        'Üniversite Mezunu': 3,
        'Yüksek Lisans': 4,
        'YÜKSEK LİSANS': 4,
        'Yüksek Lisans / Doktora': 4,
        'Yüksek Lisans / Doktara': 4,
        'Doktora': 5,
        'DOKTORA': 5
    }
    anne_calisma_mapping = {'Evet': 1,
                            'Hayır': 0,
                            'Emekli': 2
                            }
    anne_sector_mapping = {'0': 0,
                           '-': 0,
                           'Özel Sektör': 1,
                           'ÖZEL SEKTÖR': 1,
                           'Diğer': 1,
                           'DİĞER': 1,
                           'Kamu': 2,
                           'KAMU': 2
                           }
    baba_egitim_mapping = {
        'İlkokul': 0,
        'İlkokul Mezunu': 0,
        'İLKOKUL MEZUNU': 0,
        'Eğitimi yok': 0,
        'EĞİTİM YOK': 0,
        'Eğitim Yok': 0,
        'Ortaokul': 1,
        'Ortaokul Mezunu': 1,
        'ORTAOKUL MEZUNU': 1,
        'Lise': 2,
        'LİSE': 2,
        'Lise Mezunu': 2,
        'Üniversite': 3,
        'ÜNİVERSİTE': 3,
        'Üniversite Mezunu': 3,
        'Yüksek Lisans': 4,
        'YÜKSEK LİSANS': 4,
        'Yüksek Lisans / Doktora': 4,
        'Yüksek Lisans / Doktara': 4,
        'Doktora': 5,
        'DOKTORA': 5,
        '0': 0
    }
    baba_calisma_mapping = {'Evet': 1,
                            'Hayır': 0,
                            'Emekli': 2
                            }

    baba_sektor_mapping = {'0': 0,
                           '-': 0,
                           'Özel Sektör': 1,
                           'ÖZEL SEKTÖR': 1,
                           'Diğer': 1,
                           'DİĞER': 1,
                           'Kamu': 2,
                           'KAMU': 2}

    df['Cinsiyet'] = df['Cinsiyet'].map(cinsiyet_mapping)
    df['Universite Turu'] = df['Universite Turu'].map(Universite_Turu_mapping)
    df['Lise Turu'] = df['Lise Turu'].map(Lise_Turu_mapping)
    df["Anne Egitim Durumu"] = df["Anne Egitim Durumu"].map(anne_egitim_mapping)
    df['Anne Calisma Durumu'] = df['Anne Calisma Durumu'].map(anne_calisma_mapping)
    df['Anne Sektor'] = df['Anne Sektor'].map(anne_sector_mapping)
    df["Baba Egitim Durumu"] = df["Baba Egitim Durumu"].map(baba_egitim_mapping)
    df['Baba Calisma Durumu'] = df['Baba Calisma Durumu'].map(baba_calisma_mapping)
    df['Baba Sektor'] = df['Baba Sektor'].map(baba_sektor_mapping)
    df['Lise Mezuniyet Notu'] = df['Lise Mezuniyet Notu'].replace({'Not ortalaması yok': '0'})

    df['Lise Mezuniyet Notu'] = df['Lise Mezuniyet Notu'].apply(convert_to_range)

    df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].replace({
        'ORTALAMA BULUNMUYOR': '0',
        'nan': '0',
        'Hazırlığım': '0',
        'Not ortalaması yok': '0',
        'Ortalama bulunmuyor': '0'
    })

    df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].apply(convert_to_range)

    df.loc[df["Universite Kacinci Sinif"] == 'Hazırlık', "Universite Kacinci Sinif"] = "hazırlık"
    df.loc[df["Universite Kacinci Sinif"] == '0', "Universite Kacinci Sinif"] = "hazırlık"
    df.loc[df["Universite Kacinci Sinif"] == '5', "Universite Kacinci Sinif"] = "4"
    df.loc[df["Universite Kacinci Sinif"] == '6', "Universite Kacinci Sinif"] = "4"

    df.loc[df["Universite Kacinci Sinif"] == 'Tez', "Universite Kacinci Sinif"] = "Yüksek Lisans"

    df.loc[df["Ingilizce Seviyeniz?"] == '0', "Ingilizce Seviyeniz?"] = "Başlangıç"

    df.loc[((df["Spor Dalindaki Rolunuz Nedir?"] == '0') | (df["Spor Dalindaki Rolunuz Nedir?"] == 'DİĞER') |
            (df["Spor Dalindaki Rolunuz Nedir?"] == '-')),
    "Spor Dalindaki Rolunuz Nedir?"] = "diğer"

    df.loc[((df["Spor Dalindaki Rolunuz Nedir?"] == 'Lider/Kaptan') | (
            df["Spor Dalindaki Rolunuz Nedir?"] == 'KAPTAN / LİDER')),
    "Spor Dalindaki Rolunuz Nedir?"] = "kaptan"

    # Türkiye'nin en iyi 30 üniversitesinin listesi (küçük harfe dönüştürülmüş hali)
    en_iyi_20_uni = [
        'koç üniversitesi', 'hacettepe üniversitesi', 'orta doğu teknik üniversitesi', 'istanbul üniversitesi',
        'istanbul teknik üniversitesi', 'ankara üniversitesi', 'sabancı üniversitesi', 'gazi üniversitesi',
        'ege üniversitesi', 'ihsan doğramacı bilkent üniversitesi', 'istanbul üniversitesi cerrahpaşa',
        'gebze teknik üniversitesi', 'yıldız teknik üniversitesi', 'boğaziçi üniversitesi', 'marmara üniversitesi',
        'dokuz eylül üniversitesi', 'erciyes üniversitesi', 'atatürk üniversitesi', 'izmir teknoloji enstitüsü',
        'fırat üniversitesi'
    ]

    # Üniversite adlarının bulunduğu sütunun adı 'universite' olarak varsayalım
    df['Universite Adi'] = df['Universite Adi'].apply(lambda x: x if str(x).lower() in en_iyi_20_uni else 'Diğer')

    df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).applymap(
        lambda x: str(x).lower())

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


def feature_engineering(df, is_train=True):
    df['Aile Egitim Seviyesi'] = df['Anne Egitim Durumu'] + df['Baba Egitim Durumu']
    df['Aile Calisma Durumu'] = df['Anne Calisma Durumu'] + df['Baba Calisma Durumu']
    df['Aile Sosyoekonomik Durum'] = df['Aile Egitim Seviyesi'] + df['Aile Calisma Durumu']

    activity_scores = {
        'evet': 1,
        'hayır': 0
    }

    # Aktivite skorlarını hesaplama
    df['Girisimcilik Kulupleri'] = df['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'].map(activity_scores)
    df['Spor'] = df['Profesyonel Bir Spor Daliyla Mesgul musunuz?'].map(activity_scores)
    df['STK'] = df['Aktif olarak bir STK üyesi misiniz?'].map(activity_scores)
    df['Deneyim'] = df['Girisimcilikle Ilgili Deneyiminiz Var Mi?'].map(activity_scores)

    # Aktivite skorunu hesaplama
    df['Aktivite Skoru'] = (df['Girisimcilik Kulupleri'] +
                            df['Spor'] +
                            df['STK'] +
                            df['Deneyim'])
    df.drop(labels=['Girisimcilik Kulupleri', 'Spor', 'STK', 'Deneyim'], axis=1, inplace=True)

    df.dropna(inplace=True)

    liste2 = [
        'Universite Turu',
        'Burs Aliyor mu?',
        'Cinsiyet',
        'Daha Once Baska Bir Universiteden Mezun Olmus',
        'Universite Adi',
        'Lise Turu',
        'Universite Not Ortalamasi',
        'Universite Kacinci Sinif',
        'Anne Egitim Durumu',
        'Anne Calisma Durumu', 'Anne Sektor', 'Baba Egitim Durumu',
        'Baba Calisma Durumu', 'Baba Sektor',
        'Lise Mezuniyet Notu',
        'Baska Bir Kurumdan Burs Aliyor mu?',
        'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
        'Profesyonel Bir Spor Daliyla Mesgul musunuz?',
        'Spor Dalindaki Rolunuz Nedir?', 'Aktif olarak bir STK üyesi misiniz?',
        'Stk Projesine Katildiniz Mi?',
        'Girisimcilikle Ilgili Deneyiminiz Var Mi?',
        'Ingilizce Biliyor musunuz?',
        'Ingilizce Seviyeniz?',
        'Aile Egitim Seviyesi',
        'Aile Calisma Durumu',
        'Aile Sosyoekonomik Durum',
        'Aktivite Skoru'

    ]

    if is_train:
        dummies = pd.get_dummies(data=df, columns=liste2, drop_first=False)
    else:
        dummies = pd.get_dummies(data=df, columns=liste2, drop_first=False)
        # Ensure the test set has the same columns as the training set
        missing_cols = set(df_encoded_train.columns) - set(dummies.columns)
        for col in missing_cols:
            dummies[col] = 0
        dummies = dummies[df_encoded_train.columns]

    df_encoded = dummies.astype(int)
    return df, df_encoded
df_encoded_train = None




def split_train_test(df, df_encoded):
    X = df_encoded.drop(columns=['id', 'Degerlendirme Puani'])
    y = df_encoded['Degerlendirme Puani']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)
    return X_train, X_test, y_train, y_test



def cross_validate_tpot(X, y, cv=5):
    tpot_config = {
           'xgboost.XGBRegressor': {
               'n_estimators': [100, 200, 500],
               'max_depth': [3, 5, 7],
               'learning_rate': [0.01, 0.1, 0.2],
               'subsample': [0.8, 0.9, 1.0],
               'colsample_bytree': [0.8, 0.9, 1.0]
           },
           'catboost.CatBoostRegressor': {
               'iterations': [100, 200, 500],
               'depth': [3, 5, 7],
               'learning_rate': [0.01, 0.1, 0.2],
               'l2_leaf_reg': [1, 3, 5],
               'random_strength': [1, 3, 5]
           },
           'sklearn.linear_model.ElasticNet': {
               'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
               'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
           }
    }

    tpot_model = TPOTRegressor(
           generations=5,
           population_size=50,
           verbosity=2,
           random_state=42,
           config_dict=tpot_config,
           cv=cv,
           n_jobs=-1,
           max_time_mins=120,
           max_eval_time_mins=5
       )

    scores = cross_val_score(tpot_model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    return -scores.mean(), scores.std()


def test():
    global df_encoded_train
    X = yukle_ve_preprocess("train.csv")
    test = yukle_ve_preprocess("test_x.csv", is_train=False)

    print("Training set shape:", X.shape)
    print("Test set shape:", test.shape)

    # Ensure the columns match except for the target variable
    common_columns = set(X.columns).intersection(set(test.columns))
    print("Common columns:", common_columns)

    # Kategorik ve sayısal kolonları güncelle
    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns

    # Kategorik sütunları 'str' türüne dönüştür
    X[categorical_columns] = X[categorical_columns].astype(str)
    test[categorical_columns] = test[categorical_columns].astype(str)

    X, X_encoded = feature_engineering(X, is_train=True)
    df_encoded_train = X_encoded  # Store the encoded training columns
    test, test_encoded = feature_engineering(test, is_train=False)

    print("Training set shape after feature engineering:", X.shape)
    print("Test set shape after feature engineering:", test.shape)
    print("Training set encoded shape:", X_encoded.shape)
    print("Test set encoded shape:", test_encoded.shape)

    X_train, X_test, y_train, y_test = split_train_test(X, X_encoded)
    best_model = tpot(X_train, X_test, y_train, y_test)

    # Predict on the test set
    y_pred = best_model.predict(test_encoded.drop(columns=['id', 'Degerlendirme Puani']))
    print("Predictions:", y_pred)
    submission_df = pd.DataFrame({'id': test['id'], 'Degerlendirme Puani': y_pred})
    submission_df.to_csv("submission.csv", index=False)

    X, X_encoded = feature_engineering(X, is_train=True)
    y = X_encoded['Degerlendirme Puani']
    X = X_encoded.drop(columns=['id', 'Degerlendirme Puani'])

    mean_rmse, std_rmse = cross_validate_tpot(X, y)
    print(f'Cross-validated RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})')
if __name__ == '__main__':
    test()

