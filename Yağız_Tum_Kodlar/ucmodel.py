import numpy as np
import pandas as pd

df = pd.read_csv("train.csv")

liste = ['Degerlendirme Puani',
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
         'Ingilizce Biliyor musunuz?', 'Ingilizce Seviyeniz?',

         'id']
df = df.loc[:, liste]


for i in df.columns:
    print(df[i].value_counts())


#Data Preprocessing

lise_turu_mapping = {'İmam Hatip Lisesi': 0, 'Diğer': 0, 'Devlet': 0, 'Düz lise': 0,
                     'Düz Lise': 0, 'Meslek lisesi': 1, 'Meslek': 1, 'Meslek Lisesi': 1, 'Özel': 1, "Özel Lisesi": 1,
                     "Özel lisesi": 1, "Özel Lise": 1, 'Anadolu Lisesi': 2, 'Anadolu lisesi': 2, 'Fen lisesi': 3,
                     'Fen Lisesi': 3}

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
anne_calisma_mapping = {'Evet': 1, 'Hayır': 0, 'Emekli': 2}
anne_sector_mapping = {'0': 0, '-': 0, 'Özel Sektör': 1, 'ÖZEL SEKTÖR': 1, 'Diğer': 1, 'DİĞER': 1, 'Kamu': 2, 'KAMU': 2}
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
baba_calisma_mapping = {'Evet': 1, 'Hayır': 0, 'Emekli': 2}
baba_sector_mapping = {'0': 0, '-': 0, 'Özel Sektör': 1, 'ÖZEL SEKTÖR': 1, 'Diğer': 1, 'DİĞER': 1, 'Kamu': 2, 'KAMU': 2}

df['Lise Turu'] = df['Lise Turu'].map(lise_turu_mapping)
df["Anne Egitim Durumu"] = df["Anne Egitim Durumu"].map(anne_egitim_mapping)
df['Anne Calisma Durumu'] = df['Anne Calisma Durumu'].map(anne_calisma_mapping)
df['Anne Sektor'] = df['Anne Sektor'].map(anne_sector_mapping)
df["Baba Egitim Durumu"] = df["Baba Egitim Durumu"].map(baba_egitim_mapping)
df['Baba Calisma Durumu'] = df['Baba Calisma Durumu'].map(baba_calisma_mapping)
df['Baba Sektor'] = df['Baba Sektor'].map(baba_sector_mapping)

df['Lise Mezuniyet Notu'] = df['Lise Mezuniyet Notu'].replace({
    'Not ortalaması yok': '0'
})


# Değerleri aralıklara göre gruplandırma fonksiyonu
def convert_to_range(value):
    value = str(value).strip()

    # Aralıklar büyük-küçük harfe duyarlı değil, boşluklara dikkat edilerek gruplandırılıyor
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


# 'Lise Mezuniyet Notu' sütununa dönüşümü uygula
df['Lise Mezuniyet Notu'] = df['Lise Mezuniyet Notu'].apply(convert_to_range)


# Adım 2: Diğer aralıkları belirtilen gruplara dönüştürme
df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].replace({
    'ORTALAMA BULUNMUYOR': '0',
    'nan': '0',
    'Hazırlığım': '0',
    'Not ortalaması yok': '0',
    'Ortalama bulunmuyor': '0'
})

def convert_to_range(value):
    value = str(value)
    if '0 - 1.79' in value or '0-1.79' in value:
        return '0-2.5'
    elif '1.00 - 2.50' in value or '2.50 ve altı' in value or '1.80 - 2.49' in value or '2.00 - 2.50' in value :
        return '0-2.5'
    elif '2.50 - 2.99' in value  or '2.50 -3.00' in value or '3.00-2.50' in value or '2.50 - 3.00' in value:
        return '2.5-3.0'
    elif '3.00 - 3.50' in value or '3.00 - 3.49' in value or '3.00 - 4.00' in value or '3.50-3' in value :
        return '3.0-3.5'
    elif '3.50 - 4.00' in value or '4.0-3.5' in value or '4-3.5' in value:
        return '3.5-4.0'
    else:
        return '0'

df['Universite Not Ortalamasi'] = df['Universite Not Ortalamasi'].apply(convert_to_range)


df["Universite Kacinci Sinif"].value_counts()
'''
Universite Kacinci Sinif
2                21773
3                16956
4                13113
1                10260
Hazırlık          1275
5                  961
Mezun              178
6                  177
hazırlık            33
Yüksek Lisans       15
Tez                  7
0                    3
Name: count, dtype: int64
'''

df.loc[df["Universite Kacinci Sinif"] == 'Hazırlık', "Universite Kacinci Sinif"] = "hazırlık"
df.loc[df["Universite Kacinci Sinif"] == '0', "Universite Kacinci Sinif"] = "hazırlık"
df.loc[df["Universite Kacinci Sinif"] == '5', "Universite Kacinci Sinif"] = "4"
df.loc[df["Universite Kacinci Sinif"] == '6', "Universite Kacinci Sinif"] = "4"

df.loc[df["Universite Kacinci Sinif"] == 'Tez', "Universite Kacinci Sinif"] = "Yüksek Lisans"
df["Universite Kacinci Sinif"].value_counts()

'''
Universite Kacinci Sinif
2                21773
3                16956
4                14251
1                10260
hazırlık          1311
Mezun              178
Yüksek Lisans       22
Name: count, dtype: int64
'''

df.loc[df["Ingilizce Seviyeniz?"] == '0', "Ingilizce Seviyeniz?"] = "Başlangıç"

df.loc[((df["Spor Dalindaki Rolunuz Nedir?"] == '0') | (df["Spor Dalindaki Rolunuz Nedir?"] == 'DİĞER') |
       (df["Spor Dalindaki Rolunuz Nedir?"] == '-')),
       "Spor Dalindaki Rolunuz Nedir?"] = "diğer"

df.loc[((df["Spor Dalindaki Rolunuz Nedir?"] == 'Lider/Kaptan') | (df["Spor Dalindaki Rolunuz Nedir?"] == 'KAPTAN / LİDER') ),
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


# Cinsiyet sütunu hariç tüm sütunlarda 'kadın' ve 'erkek' değerlerini kaldır
df.loc[:, df.columns != 'cinsiyet'] = df.loc[:, df.columns != 'cinsiyet'].applymap(lambda x: '' if x in ['kadın', 'erkek'] else x)


df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).applymap(lambda x: str(x).lower())


df.loc[df["Universite Turu"]=="nan","Universite Turu"] = "devlet"
df.loc[df["Daha Once Baska Bir Universiteden Mezun Olmus"]=="nan","Daha Once Baska Bir Universiteden Mezun Olmus"] = "hayır"
df.loc[df["Baska Bir Kurumdan Burs Aliyor mu?"]=="nan","Baska Bir Kurumdan Burs Aliyor mu?"] = "hayır"
df.loc[df["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"]=="nan","Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = "hayır"
df.loc[df["Profesyonel Bir Spor Daliyla Mesgul musunuz?"]=="nan","Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = "hayır"
df.loc[df["Spor Dalindaki Rolunuz Nedir?"]=="nan","Spor Dalindaki Rolunuz Nedir?"] = "diğer"
df.loc[df["Aktif olarak bir STK üyesi misiniz?"]=="nan","Aktif olarak bir STK üyesi misiniz?"] = "hayır"
df.loc[df["Stk Projesine Katildiniz Mi?"]=="nan","Stk Projesine Katildiniz Mi?"] = "hayır"
df.loc[df["Girisimcilikle Ilgili Deneyiminiz Var Mi?"]=="nan","Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = "hayır"
df.loc[df["Ingilizce Biliyor musunuz?"]=="nan","Ingilizce Biliyor musunuz?"] = "hayır"
df.loc[df["Ingilizce Seviyeniz?"]=="nan","Ingilizce Seviyeniz?"] = "başlangıç"
df.loc[df["Cinsiyet"]=="nan","Cinsiyet"] = "erkek"
df.loc[df["Cinsiyet"]=="belirtmek istemiyorum","Cinsiyet"] = "kadın"

#Feature Engineering

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

df.drop(labels=['Girisimcilik Kulupleri','Spor','STK','Deneyim'],axis=1,inplace=True)

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

numeric_columns = df[liste2].select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].astype(str)

dummies = pd.get_dummies(data=df,columns=liste2,drop_first=False)


df_encoded = dummies.astype(int)
df_encoded.head()


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Özellikler ve hedef değişkenlerin ayrılması
X = df_encoded.drop(columns=['id', 'Degerlendirme Puani'])
y = df_encoded['Degerlendirme Puani']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear Regression Modeli
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f'Linear Regression RMSE: {rmse_linear}')

# Random Forest Hiperparametre Optimizasyonu
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=1)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'Random Forest Best RMSE: {rmse_rf}')
print('Best parameters for Random Forest:', rf_grid_search.best_params_)

# Özelliklerin önem derecelerinin hesaplanması (Random Forest)
feature_importances_rf = best_rf_model.feature_importances_
importance_df_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances_rf
}).sort_values(by='Importance', ascending=False)
print('Random Forest Feature Importances:\n', importance_df_rf)

# XGBoost Hiperparametre Optimizasyonu
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=1)
xgb_grid_search.fit(X_train, y_train)
best_xgb_model = xgb_grid_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'XGBoost Best RMSE: {rmse_xgb}')
print('Best parameters for XGBoost:', xgb_grid_search.best_params_)

# Özelliklerin önem derecelerinin hesaplanması (XGBoost)
feature_importances_xgb = best_xgb_model.feature_importances_
importance_df_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances_xgb
}).sort_values(by='Importance', ascending=False)
print('XGBoost Feature Importances:\n', importance_df_xgb)

# En iyi modeli seçmek
best_rmse = min(rmse_linear, rmse_rf, rmse_xgb)
print(f'En iyi RMSE: {best_rmse}')

'''
Linear Regression RMSE: 6.129980791402805
Random Forest Best RMSE: 5.472436439046333
Best parameters for Random Forest: {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 150}
Random Forest Feature Importances:
                                                Feature  Importance
72   Girisimcilik Kulupleri Tarzi Bir Kulube Uye mi...    0.190446
71   Girisimcilik Kulupleri Tarzi Bir Kulube Uye mi...    0.149713
65                               Lise Mezuniyet Notu_0    0.101908
80           Aktif olarak bir STK üyesi misiniz?_hayır    0.061807
79            Aktif olarak bir STK üyesi misiniz?_evet    0.058786
..                                                 ...         ...
17            Universite Adi_gebze teknik üniversitesi    0.000009
100                           Aile Egitim Seviyesi_9.0    0.000006
92                           Aile Egitim Seviyesi_10.0    0.000004
110                      Aile Sosyoekonomik Durum_12.0    0.000003
40              Universite Kacinci Sinif_yüksek lisans    0.000000

[124 rows x 2 columns]
XGBoost Best RMSE: 5.177087731436845
Best parameters for XGBoost: {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 150, 'subsample': 0.9}
XGBoost Feature Importances:
                                                Feature  Importance
71   Girisimcilik Kulupleri Tarzi Bir Kulube Uye mi...    0.253968
74   Profesyonel Bir Spor Daliyla Mesgul musunuz?_h...    0.143934
65                               Lise Mezuniyet Notu_0    0.090222
79            Aktif olarak bir STK üyesi misiniz?_evet    0.088003
81                   Stk Projesine Katildiniz Mi?_evet    0.059604
..                                                 ...         ...
13                     Universite Adi_ege üniversitesi    0.000466
105                            Aile Calisma Durumu_4.0    0.000389
39                        Universite Kacinci Sinif_nan    0.000312
110                      Aile Sosyoekonomik Durum_12.0    0.000303
40              Universite Kacinci Sinif_yüksek lisans    0.000206

[124 rows x 2 columns]
En iyi RMSE: 5.177087731436845
'''