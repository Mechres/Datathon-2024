import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def yukle_ve_preprocess(dosya_yolu):
    print(f"Dosya yükleniyor: {dosya_yolu}")
    df = pd.read_csv(dosya_yolu)
    print(f"Veri seti boyutu: {df.shape}")

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
    if 'Degerlendirme Puani' in df.columns:
        tutulacak_sutunlar.append('Degerlendirme Puani')
    df = df.loc[:, tutulacak_sutunlar]

    # Mappings
    cinsiyet_mapping = {'Erkek': 0, 'Kadın': 1, 'Belirtmek istemiyorum': 3}
    Universite_Turu_mapping = {'Devlet': 0, 'Özel': 1}
    Lise_Turu_mapping = {
        'İmam Hatip Lisesi': 0, 'Diğer': 0, 'Devlet': 0, 'Düz lise': 0, 'Düz Lise': 0,
        'Meslek lisesi': 1, 'Meslek': 1, 'Meslek Lisesi': 1, 'Özel': 1, "Özel Lisesi": 1,
        "Özel lisesi": 1, "Özel Lise": 1, 'Anadolu Lisesi': 2, 'Anadolu lisesi': 2,
        'Fen lisesi': 3, 'Fen Lisesi': 3
    }
    anne_egitim_mapping = {
        'İlkokul': 0, 'İlkokul Mezunu': 0, 'İLKOKUL MEZUNU': 0, 'Eğitimi yok': 0, 'EĞİTİM YOK': 0,
        'Eğitim Yok': 0, 'Ortaokul': 1, 'Ortaokul Mezunu': 1, 'ORTAOKUL MEZUNU': 1, 'Lise': 2,
        'LİSE': 2, 'Lise Mezunu': 2, 'Üniversite': 3, 'ÜNİVERSİTE': 3, 'Üniversite Mezunu': 3,
        'Yüksek Lisans': 4, 'YÜKSEK LİSANS': 4, 'Yüksek Lisans / Doktora': 4, 'Yüksek Lisans / Doktara': 4,
        'Doktora': 5, 'DOKTORA': 5
    }
    anne_calisma_mapping = {'Evet': 1, 'Hayır': 0, 'Emekli': 2}
    anne_sector_mapping = {'0': 0, '-': 0, 'Özel Sektör': 1, 'ÖZEL SEKTÖR': 1, 'Diğer': 1, 'DİĞER': 1, 'Kamu': 2,
                           'KAMU': 2}
    baba_egitim_mapping = anne_egitim_mapping.copy()
    baba_egitim_mapping['0'] = 0
    baba_calisma_mapping = anne_calisma_mapping
    baba_sektor_mapping = anne_sector_mapping

    # Apply mappings
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
        'ORTALAMA BULUNMUYOR': '0', 'nan': '0', 'Hazırlığım': '0',
        'Not ortalaması yok': '0', 'Ortalama bulunmuyor': '0'
    })
    df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].apply(convert_to_range)

    df.loc[df["Universite Kacinci Sinif"] == 'Hazırlık', "Universite Kacinci Sinif"] = "hazırlık"
    df.loc[df["Universite Kacinci Sinif"] == '0', "Universite Kacinci Sinif"] = "hazırlık"
    df.loc[df["Universite Kacinci Sinif"].isin(['5', '6']), "Universite Kacinci Sinif"] = "4"
    df.loc[df["Universite Kacinci Sinif"] == 'Tez', "Universite Kacinci Sinif"] = "Yüksek Lisans"

    df.loc[df["Ingilizce Seviyeniz?"] == '0', "Ingilizce Seviyeniz?"] = "Başlangıç"

    df.loc[df["Spor Dalindaki Rolunuz Nedir?"].isin(['0', 'DİĞER', '-']), "Spor Dalindaki Rolunuz Nedir?"] = "diğer"
    df.loc[df["Spor Dalindaki Rolunuz Nedir?"].isin(
        ['Lider/Kaptan', 'KAPTAN / LİDER']), "Spor Dalindaki Rolunuz Nedir?"] = "kaptan"

    en_iyi_20_uni = [
        'koç üniversitesi', 'hacettepe üniversitesi', 'orta doğu teknik üniversitesi', 'istanbul üniversitesi',
        'istanbul teknik üniversitesi', 'ankara üniversitesi', 'sabancı üniversitesi', 'gazi üniversitesi',
        'ege üniversitesi', 'ihsan doğramacı bilkent üniversitesi', 'istanbul üniversitesi cerrahpaşa',
        'gebze teknik üniversitesi', 'yıldız teknik üniversitesi', 'boğaziçi üniversitesi', 'marmara üniversitesi',
        'dokuz eylül üniversitesi', 'erciyes üniversitesi', 'atatürk üniversitesi', 'izmir teknoloji enstitüsü',
        'fırat üniversitesi'
    ]

    df['Universite Adi'] = df['Universite Adi'].apply(lambda x: x if str(x).lower() in en_iyi_20_uni else 'Diğer')

    df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).applymap(
        lambda x: str(x).lower())

    default_values = {
        "Daha Once Baska Bir Universiteden Mezun Olmus": "hayır",
        "Baska Bir Kurumdan Burs Aliyor mu?": "hayır",
        "Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?": "hayır",
        "Profesyonel Bir Spor Daliyla Mesgul musunuz?": "hayır",
        "Spor Dalindaki Rolunuz Nedir?": "diğer",
        "Aktif olarak bir STK üyesi misiniz?": "hayır",
        "Stk Projesine Katildiniz Mi?": "hayır",
        "Girisimcilikle Ilgili Deneyiminiz Var Mi?": "hayır",
        "Ingilizce Biliyor musunuz?": "hayır",
        "Ingilizce Seviyeniz?": "başlangıç"
    }

    for col, default_value in default_values.items():
        df.loc[df[col] == "nan", col] = default_value

    print("Preprocessing tamamlandı.")
    print(f"Preprocessed veri seti boyutu: {df.shape}")
    return df


def convert_to_range(value):
    value = str(value).strip().lower()
    if any(x in value for x in ['0 - 24', '0 - 25', '25 - 49', '44-0', '25 - 50', '54-45']):
        return '0-50'
    elif any(x in value for x in ['50 - 74', '50 - 75', '54-45', '69-55', '84-70', '3.00-2.50']):
        return '50-75'
    elif any(x in value for x in ['75 - 100', '100-85', '3.00 - 4.00', '3.50-3.00', '3.50-3', '4.00-3.50']):
        return '75-100'
    elif '2.50 ve altı' in value:
        return '0-50'
    else:
        return '0'


def feature_engineering(df):
    print(f"Feature engineering başlıyor... Başlangıç boyutu: {df.shape}")

    df['Aile Egitim Seviyesi'] = df['Anne Egitim Durumu'] + df['Baba Egitim Durumu']
    print(f"Aile Egitim Seviyesi eklendi. Yeni boyut: {df.shape}")
    df['Aile Calisma Durumu'] = df['Anne Calisma Durumu'] + df['Baba Calisma Durumu']
    print(f"Aile Calisma Durumu eklendi. Yeni boyut: {df.shape}")
    df['Aile Sosyoekonomik Durum'] = df['Aile Egitim Seviyesi'] + df['Aile Calisma Durumu']
    print(f"Aile Sosyoekonomik Durum eklendi. Yeni boyut: {df.shape}")
    activity_scores = {'evet': 1, 'hayır': 0}

    for col in ['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
                'Profesyonel Bir Spor Daliyla Mesgul musunuz?',
                'Aktif olarak bir STK üyesi misiniz?',
                'Girisimcilikle Ilgili Deneyiminiz Var Mi?']:
        df[col] = df[col].map(activity_scores)
    print(f"df[col] = df[col].map(activity_scores). Yeni boyut: {df.shape}")

    df['Aktivite Skoru'] = df['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'] + \
                           df['Profesyonel Bir Spor Daliyla Mesgul musunuz?'] + \
                           df['Aktif olarak bir STK üyesi misiniz?'] + \
                           df['Girisimcilikle Ilgili Deneyiminiz Var Mi?']
    print(f"Aktivite Skoru eklendi. Yeni boyut: {df.shape}")

    df.drop(labels=['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
                    'Profesyonel Bir Spor Daliyla Mesgul musunuz?',
                    'Aktif olarak bir STK üyesi misiniz?',
                    'Girisimcilikle Ilgili Deneyiminiz Var Mi?'], axis=1, inplace=True)
    print(f"df.drop(labels. Yeni boyut: {df.shape}")

    df.dropna(inplace=True)

    categorical_columns = [
        'Universite Turu', 'Burs Aliyor mu?', 'Cinsiyet',
        'Daha Once Baska Bir Universiteden Mezun Olmus', 'Universite Adi',
        'Lise Turu', 'Universite Not Ortalamasi', 'Universite Kacinci Sinif',
        'Anne Egitim Durumu', 'Anne Calisma Durumu', 'Anne Sektor',
        'Baba Egitim Durumu', 'Baba Calisma Durumu', 'Baba Sektor',
        'Lise Mezuniyet Notu', 'Baska Bir Kurumdan Burs Aliyor mu?',
        'Spor Dalindaki Rolunuz Nedir?', 'Stk Projesine Katildiniz Mi?',
        'Ingilizce Biliyor musunuz?', 'Ingilizce Seviyeniz?'
    ]

    numeric_columns = df[categorical_columns].select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].astype(str)
    print(f"numeric_columns. Yeni boyut: {df.shape}")

    dummies = pd.get_dummies(data=df, columns=categorical_columns, drop_first=False)
    df_encoded = dummies.astype(int)
    print(f"dummies. Yeni boyut: {df.shape}")

    print(f"Feature engineering tamamlandı. Son boyut: {df.shape}")
    return df, df_encoded

def split_train_test(df, df_encoded):
    print("Veri seti bölünüyor...")
    X = df_encoded.drop(columns=['id', 'Degerlendirme Puani'])
    y = df_encoded['Degerlendirme Puani']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def xgboost_model(X_train, y_train):
    print("XGBoost modeli eğitiliyor...")
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

    xgb_param_dist = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_dist,
                                           n_iter=10, cv=3, scoring='neg_root_mean_squared_error',
                                           n_jobs=-1, random_state=42, verbose=2)
    xgb_random_search.fit(X_train, y_train)

    best_xgb_model = xgb_random_search.best_estimator_
    print('En iyi XGBoost parametreleri:', xgb_random_search.best_params_)
    return best_xgb_model


if __name__ == '__main__':
    try:
        print("Program başlıyor...")

        # Eğitim verisini yükle ve işle
        train_df = yukle_ve_preprocess("train.csv")
        train_df, train_df_encoded = feature_engineering(train_df)
        X_train, X_test, y_train, y_test = split_train_test(train_df, train_df_encoded)

        # XGBoost modelini eğit
        best_xgb_model = xgboost_model(X_train, y_train)

        # Test seti üzerinde değerlendir
        y_pred_xgb = best_xgb_model.predict(X_test)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        print(f'XGBoost Test RMSE: {rmse_xgb}')

        # Test verisini yükle ve işle
        print("Test verisi yükleniyor ve işleniyor...")
        test_df = yukle_ve_preprocess("test_x.csv")
        test_df, test_df_encoded = feature_engineering(test_df)
        print(f"Test verisi boyutu (encoded): {test_df_encoded.shape}")
        for col in X_train.columns:
            if col not in test_df_encoded.columns:
                test_df_encoded[col] = 0
        test_df_encoded = test_df_encoded[X_train.columns]

        # Test verisinin eğitim verisiyle aynı sütunlara sahip olduğundan emin ol
        print("Test verisi sütunları kontrol ediliyor...")
        missing_cols = set(X_train.columns) - set(test_df_encoded.columns)
        for col in missing_cols:
            test_df_encoded[col] = 0
        test_df_encoded = test_df_encoded[X_train.columns]

        print(f"Test verisi boyutu: {test_df_encoded.shape}")
        print(f"Eğitim verisi sütunları: {X_train.columns}")
        print(f"Test verisi sütunları: {test_df_encoded.columns}")

        # Test verisi üzerinde tahmin yap
        print("Test verisi üzerinde tahmin yapılıyor...")
        test_predictions = best_xgb_model.predict(test_df_encoded)

        print(f"Tahmin sayısı: {len(test_predictions)}")
        print(f"İlk 5 tahmin: {test_predictions[:5]}")

        # Submission dosyasını oluştur
        print("Submission dosyası oluşturuluyor...")
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Degerlendirme Puani': test_predictions
        })
        print(f"Submission boyutu: {submission.shape}")
        print(f"Submission ilk 5 satır:\n{submission.head()}")

        submission.to_csv('submission.csv', index=False)
        print("Tahminler 'submission.csv' dosyasına kaydedildi.")

    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
        print("Hata detayları:")
        import traceback

        traceback.print_exc()