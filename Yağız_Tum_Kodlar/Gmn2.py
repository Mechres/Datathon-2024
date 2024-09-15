import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
from scipy.stats import uniform, randint


# 14.09.2024 Rmse: 9.***

def eksik_veri_yonetimi(df):
    """ Gelişmiş yöntemlerle eksik değerleri doldurur. """
    # KNN Imputer veya IterativeImputer kullanabilirsiniz
    imputer = KNNImputer(n_neighbors=5)  # Örnek olarak KNN Imputer
    df_imputed = imputer.fit_transform(df)
    return pd.DataFrame(df_imputed, columns=df.columns)


def aykiri_deger_islemi(df):
    """ Aykırı değerleri tespit eder ve işler. """
    # IQR yöntemi veya diğer yöntemlerle aykırı değerleri tespit edebilir ve kaldırabilir veya dönüştürebilirsiniz
    # Örnek olarak, winsorize işlemi uygulanabilir
    from scipy.stats.mstats import winsorize
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = winsorize(df[col], limits=[0.05, 0.05])  # Alt ve üst %5'lik dilimi kırp
    return df


def ozellik_secimi(X_train, y_train):
    """ Önemli özellikleri seçer. """
    # RFE veya SelectFromModel kullanabilirsiniz
    estimator = XGBRegressor(objective='reg:squarederror', random_state=5)
    selector = RFE(estimator, n_features_to_select=10, step=1)  # Örnek olarak RFE
    selector = selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.support_]
    return selected_features


def ozellik_donusumleri(df):
    """ Özelliklerin dağılımını dönüştürür. """
    # Log dönüşümü, Box-Cox dönüşümü veya standartlaştırma uygulayabilirsiniz
    scaler = StandardScaler()
    transformer = PowerTransformer(method='yeo-johnson')  # Örnek olarak Yeo-Johnson dönüşümü
    df_transformed = scaler.fit_transform(df)
    df_transformed = transformer.fit_transform(df_transformed)
    return pd.DataFrame(df_transformed, columns=df.columns)


def split_train_test(df, df_encoded):
    """ Veriyi eğitim ve test setlerine böler, veri sızıntısını önler. """
    # Özellik mühendisliği ve veri bölme işlemlerini aynı pipeline içinde yaparak veri sızıntısını önleyebilirsiniz.
    X = df_encoded.drop(columns=['id', 'Degerlendirme Puani'])
    y = df_encoded['Degerlendirme Puani']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)
    return X_train, X_test, y_train, y_test


from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def xgboost_with_hyperparameter_tuning(X_train, y_train, gridsearch=False):
    """ XGBoost modelini eğitir ve opsiyonel olarak GridSearch veya RandomizedSearch ile hiperparametreleri ayarlar. """
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=5)

    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'alpha': [0, 0.01, 0.1],
        'lambda': [0.01, 0.1, 1]
    }

    if gridsearch:
        # GridSearchCV kullanarak hiperparametre ayarlaması
        search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid,
                              cv=10, scoring='neg_root_mean_squared_error',
                              n_jobs=-1, verbose=1)
    else:
        # RandomizedSearchCV kullanarak hiperparametre ayarlaması
        search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_grid,
                                    n_iter=10, cv=10, scoring='neg_root_mean_squared_error',
                                    n_jobs=-1, random_state=5, verbose=1)

    # Modeli eğit ve en iyi parametreleri bul
    for _ in tqdm(range(search.n_iter), desc="Hiperparametre Ayarlaması"):
        search.fit(X_train, y_train)

    # Denenen tüm parametreleri ve skorlarını yazdır
    results = search.cv_results_
    for i in range(len(results['params'])):
        params = results['params'][i]
        mean_score = results['mean_test_score'][i]
        print(f"Trial {i + 1}: Params: {params}, Mean Test Score: {mean_score}")

    best_xgb_model = search.best_estimator_
    print('Best parameters for XGBoost:', search.best_params_)
    return best_xgb_model


def evaluate_model(model, X_train, y_train):
    """ Modeli çapraz doğrulama ile değerlendirir. """
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores  # Skorları pozitif RMSE değerlerine dönüştür
    print(f'Cross-validation RMSE scores: {rmse_scores}')
    print(f'Mean RMSE: {np.mean(rmse_scores)} +/- {np.std(rmse_scores)}')


def topluluk_modeli(X_train, y_train):
    """ Topluluk modeli oluşturur. """

    # Modellerin tanımlanması
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=5)
    rf_model = RandomForestRegressor(random_state=5)
    catboost_model = CatBoostRegressor(random_state=5, verbose=0)
    svr_model = SVR()

    estimators = [
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('catboost', catboost_model),
        ('svr', svr_model)
    ]

    # Veriyi eğitim ve doğrulama kümelerine ayırın
    X_train_stacking, X_val_stacking, y_train_stacking, y_val_stacking = train_test_split(X_train, y_train,
                                                                                          test_size=0.2,
                                                                                          random_state=42)

    # Final model (meta-estimator)
    final_estimator = XGBRegressor(objective='reg:squarederror', random_state=5)

    # Stacking regressor oluşturulması
    ensemble_model = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)

    # Topluluk modelini eğitin
    ensemble_model.fit(X_train, y_train)

    return ensemble_model


def feature_engineering(df, is_train=True):
    """Yeni özellikler oluşturur."""

    # Add new features
    #df['Age'] = df['Basvuru Yili'] - pd.to_datetime(df['Dogum Tarihi']).dt.year
    #df['Is_Local_Student'] = (df['Ikametgah Sehri'] == df['Lise Sehir']).astype(int)
    df['Family_Education_Gap'] = df['Baba Egitim Durumu'] - df['Anne Egitim Durumu']
    df['Total_Clubs'] = (df['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'] == 'evet').astype(int) + \
                        (df['Profesyonel Bir Spor Daliyla Mesgul musunuz?'] == 'evet').astype(int) + \
                        (df['Aktif olarak bir STK üyesi misiniz?'] == 'evet').astype(int)

    # Create interaction features
    #df['Uni_Grade_x_Year'] = df['Universite Not Ortalamasi'] * df['Universite Kacinci Sinif']
    df['Family_Education_x_Work'] = df['Family_Education_Gap'] * ((df['Anne Calisma Durumu'] == 'Evet').astype(int) +
                                                                  (df['Baba Calisma Durumu'] == 'Evet').astype(int))
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
    return df
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


def advanced_imputation(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    # For numeric features, use KNN imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    df[numeric_features] = knn_imputer.fit_transform(df[numeric_features])

    # For categorical features, use mode imputation
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = mode_imputer.fit_transform(df[categorical_features])

    return df


def advanced_feature_selection(X, y, n_features=30):
    # Use mutual information for feature selection
    mi_scores = mutual_info_regression(X, y)
    mi_features = pd.Series(mi_scores, index=X.columns).nlargest(n_features).index.tolist()

    # Use Lasso for feature selection
    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)
    lasso_features = X.columns[np.abs(lasso.coef_) > 0].tolist()

    # Combine features from both methods
    selected_features = list(set(mi_features + lasso_features))

    return selected_features


def create_advanced_ensemble(X_train, y_train):
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('catboost', CatBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=0)),
        ('svr', SVR(kernel='rbf')),
        ('lasso', Lasso(alpha=0.01)),
        ('ridge', Ridge(alpha=1.0)),
        ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.5))
    ]

    meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    ensemble = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )

    return ensemble


def hyper_parameter_tuning(model, param_grid, X_train, y_train):
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def yukle_ve_preprocess(dosya_yolu, is_train=True):
    df = pd.read_csv(dosya_yolu)

    if is_train:
        tutulacak_sutunlar = [
            'Basvuru Yili',  'Degerlendirme Puani', 'Universite Turu', 'Burs Aliyor mu?', 'Cinsiyet',
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

    # Eksik değerleri doldurma (sayısal sütunlar için medyan, kategorik sütunlar için mod)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:  # Sayısal sütun
            df[col] = df[col].fillna(df[col].median())
        else:  # Kategorik sütun
            df[col] = df[col].fillna(df[col].mode()[0])

    # Mappingler
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


def preprocess_all(train_dosya, test_dosya):
    """ Train ve test veri setlerini yükler, önişlemden geçirir ve özellik mühendisliği uygular. """
    global df_encoded_train
    with tqdm(total=2, desc="Veri Önişleme") as pbar:  # Toplam 2 işlem (train ve test)
        X = yukle_ve_preprocess(train_dosya)
        pbar.update(1)  # Train veri seti işlendi
        test = yukle_ve_preprocess(test_dosya, is_train=False)
        pbar.update(1)  # Test veri seti işlendi

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
    print("Test set encoded shape:", test_encoded.shape)  # Parantez eklendi

    return X, X_encoded, df_encoded_train, test, test_encoded


def main():
    train_file = "train.csv"
    test_file = "test_x.csv"

    # Load and preprocess data
    train_df = yukle_ve_preprocess(train_file)
    test_df = yukle_ve_preprocess(test_file, is_train=False)

    # Feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Advanced imputation
    train_df = advanced_imputation(train_df)
    test_df = advanced_imputation(test_df)

    # Split features and target
    X = train_df.drop(['Degerlendirme Puani', 'id'], axis=1)
    y = train_df['Degerlendirme Puani']

    # Advanced feature selection
    selected_features = advanced_feature_selection(X, y)
    X = X[selected_features]
    test_df = test_df[selected_features]

    # Split train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train advanced ensemble
    ensemble = create_advanced_ensemble(X_train, y_train)
    ensemble.fit(X_train, y_train)

    # Hyperparameter tuning for the best performing model in the ensemble
    best_model = ensemble.named_estimators_[ensemble.final_estimator_.feature_importances_.argmax()]
    param_grid = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }
    best_model_tuned = hyper_parameter_tuning(best_model, param_grid, X_train, y_train)

    # Make predictions
    y_pred = ensemble.predict(X_val)
    y_pred_tuned = best_model_tuned.predict(X_val)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_tuned = np.sqrt(mean_squared_error(y_val, y_pred_tuned))

    print(f"Ensemble RMSE: {rmse}")
    print(f"Tuned Best Model RMSE: {rmse_tuned}")

    # Make predictions on test set
    test_pred = ensemble.predict(test_df)
    test_pred_tuned = best_model_tuned.predict(test_df)

    # Create submission file
    submission_df = pd.DataFrame({'id': test_df['id'], 'Degerlendirme Puani': (test_pred + test_pred_tuned) / 2})
    submission_df.to_csv("improved_submission.csv", index=False)
    print("Predictions saved to improved_submission.csv")


if __name__ == '__main__':
    main()