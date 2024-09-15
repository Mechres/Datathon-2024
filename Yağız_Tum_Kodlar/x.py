import pandas as pd
import numpy as np
import re
from datetime import datetime

from pandas.core.computation.parsing import clean_column_name
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the training data with low_memory=False to handle mixed dtypes
train = pd.read_csv("train.csv", low_memory=False)
print(train.columns)
# Define the columns based on the provided data structure
train.columns = [
    'Basvuru_Yili', 'Degerlendirme_Puani', 'Cinsiyet', 'Dogum_Tarihi', 'Dogum_Yeri', 'Ikametgah_Sehri',
    'Universite_Adi', 'Universite_Turu', 'Burs_Yuzdesi', 'Burs_Aliyor_mu', 'Bolum', 'Universite_Kacinci_Sinif',
    'Universite_Not_Ortalamasi', 'Baska_Universiteden_Mezun', 'Lise_Adi', 'Lise_Adi_Diger', 'Lise_Sehir',
    'Lise_Turu', 'Lise_Bolumu', 'Lise_Bolum_Diger', 'Lise_Mezuniyet_Notu', 'Baska_Kurumdan_Burs_Aliyor_mu',
    'Burs_Aldigi_Baska_Kurum', 'Burs_Miktari', 'Anne_Egitim_Durumu', 'Anne_Calisma_Durumu', 'Anne_Sektor',
    'Baba_Egitim_Durumu', 'Baba_Calisma_Durumu', 'Baba_Sektor', 'Kardes_Sayisi', 'Girisimcilik_Kulub_Uye_mi',
    'Kulup_Ismi', 'Spor_Dali_Mesgul_mu', 'Spor_Rolu', 'STK_Uyesi_mi', 'STK_Ismi', 'STK_Projesi_Katilim',
    'Girisimcilik_Deneyimi', 'Girisimcilik_Deneyim_Aciklama', 'Ingilizce_Biliyor_mu', 'Ingilizce_Seviyesi',
    'Mezun_Olunan_Universite', 'id'
]

# Convert date of birth to datetime and calculate age
train['Dogum_Tarihi'] = pd.to_datetime(train['Dogum_Tarihi'], errors='coerce')
today = pd.to_datetime(datetime.now())
train['Yas'] = (today - train['Dogum_Tarihi']).dt.days / 365.25

sehirler = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Aksaray", "Amasya", "Ankara", "Antalya",
    "Ardahan", "Artvin", "Aydın", "Balıkesir", "Bartın", "Batman", "Bayburt", "Bilecik", "Bingöl",
    "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır",
    "Düzce", "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane",
    "Hakkari", "Hatay", "Iğdır", "Isparta", "İstanbul", "İzmir", "Kahramanmaraş", "Karabük", "Karaman",
    "Kars", "Kastamonu", "Kayseri", "Kilis", "Kırıkkale", "Kırklareli", "Kırşehir", "Kocaeli", "Konya",
    "Kütahya", "Malatya", "Manisa", "Mardin", "Mersin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu",
    "Osmaniye", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Şanlıurfa", "Şırnak", "Tekirdağ",
    "Tokat", "Trabzon", "Tunceli", "Uşak", "Van", "Yalova", "Yozgat", "Zonguldak"
]


def temizle_sehir(sehir):
    if pd.isna(sehir):
        return None
    if isinstance(sehir, str):
        sehir = sehir.lower().strip()
        for il in sehirler:
            if il.lower() in sehir:
                return il
    return None


# Apply the city cleaning function
train["Dogum_Yeri_Temizlenmiş"] = train["Dogum_Yeri"].apply(temizle_sehir)
train["Ikametgah_Sehri_Temizlenmiş"] = train["Ikametgah_Sehri"].apply(temizle_sehir)
train["Lise_Sehir"] = train["Lise_Sehir"].apply(temizle_sehir)

# Fill missing values in 'Burs Aldigi Baska Kurum' and create burs_kategorisi
train["Burs_Aldigi_Baska_Kurum"] = train["Burs_Aldigi_Baska_Kurum"].fillna("-").str.lower()
train["burs_kategorisi"] = train["Burs_Aldigi_Baska_Kurum"].apply(lambda x: 0 if x == "-" else (1 if "kyk" in x else 2))
train.drop(columns=['Burs_Aldigi_Baska_Kurum'], inplace=True)

# Handle 'Bölüm' encoding (rename to match the updated column name)
top_40_categories = train["Bolum"].value_counts().head(40).index
train["Bolum"] = train["Bolum"].apply(lambda x: x if x in top_40_categories else "Other")
dummies = pd.get_dummies(train["Bolum"], prefix="Bolum")
train["Bolum_encoded"] = dummies.T.groupby(dummies.columns).sum().idxmax(axis=1)
train.drop("Bolum", axis=1, inplace=True)


# Other preprocessing (e.g., extracting numbers, filling missing values, mappings)
def extract_numbers(text):
    if isinstance(text, str):
        numbers = re.findall(r'\d+', text)
        return int(max(numbers, key=int)) if numbers else 0
    return 0


if 'Degerlendirme Puani' in train.columns:
    train.rename(columns={'Degerlendirme Puani': 'Degerlendirme_Puani'}, inplace=True)

train['Burs_Miktari'] = train['Burs_Miktari'].apply(extract_numbers)

train['Degerlendirme_Puani'] = train['Degerlendirme_Puani'].fillna(0)
train['Dogum_Yeri_Temizlenmiş'] = train['Dogum_Yeri_Temizlenmiş'].fillna(0)
train['Ikametgah_Sehri_Temizlenmiş'] = train['Ikametgah_Sehri_Temizlenmiş'].fillna(0)

# Mappings
cinsiyet_mapping = {'Erkek': 1, 'ERKEK': 1, 'Kadın': 2, 'Belirtmek istemiyorum': 0}
uni_mapping = {'Devlet': 1, 'DEVLET': 1, 'Özel': 0, 'ÖZEL': 0}
burs_mapping = {'evet': 1, 'Evet': 1, 'EVET': 1, 'Hayır': 0, 'hayır': 0}
uni_sinif_mapping = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 'Yüksek Lisans': 7, 'tez': 7, 'Mezun': 0,
                     "Hazırlık": 0, "hazırlık": 0, "0": 0}
uni_not_mapping = {'ORTALAMA BULUNMUYOR': 0, 'Not ortalaması yok': 0, 'Ortalama bulunmuyor': 0, 'Hazırlığım': 0,
                   '0 - 1.79': 0, '1.00 - 2.50': 1, '1.80 - 2.49': 1, '2.50 ve altı': 1, '2.00 - 2.50': 1,
                   "3.00-2.50": 2,
                   "2.50 - 3.00": 2, "2.50 - 2.99": 2, '2.50 -3.00': 2, '3.00 - 3.50': 3, '3.50-3': 3, '3.00 - 3.49': 3,
                   '3.00 - 4.00': 3, '3.50 - 4.00': 4, '4-3.5': 4, '4.0-3.5': 4}
mezun_mapping = {'Evet': 1, 'Hayır': 0}
lise_turu_mapping = {'İmam Hatip Lisesi': 0, 'Diğer': 0, 'Devlet': 0, 'Düz lise': 0,
                     'Düz Lise': 0, 'Meslek lisesi': 1, 'Meslek': 1, 'Meslek Lisesi': 1, 'Özel': 1, "Özel Lisesi": 1,
                     "Özel lisesi": 1, "Özel Lise": 1, 'Anadolu Lisesi': 2, 'Anadolu lisesi': 2, 'Fen lisesi': 3,
                     'Fen Lisesi': 3}
lise_not_mapping = {
    '100 - 75': 5,
    '84 - 70': 4,
    '100 - 85': 5,
    '4.00 - 3.50': 5,
    '50 - 75': 3,
    '3.00 - 4.00': 4,
    '3.50 - 3.00': 4,
    '3.50 - 3': 4,
    '69 - 55': 3,
    '3.00 - 2.50': 3,
    '50 - 74': 3,
    '2.50 ve altı': 2,
    '54 - 45': 2,
    '25 - 50': 1,
    'Not ortalaması yok': 0,
    '44 - 0': 0,
    '0 - 25': 0,
    '25 - 49': 1,
    '0 - 24': 0
}
burslu_mapping = {'Evet': 1, 'Hayır': 0}
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
kardes_sayisi_mapping = {
    'Kardeş Sayısı 1 Ek Bilgi Aile Hk. Anne Vefat': 100,
    0.0: 0,
    1.0: 1,
    2.0: 2,
    3.0: 3,
    4.0: 4,
    5.0: 5,
    6.0: 6,
    7.0: 7,
    8.0: 8,
    9.0: 9,
    10.0: 10,
    11.0: 11,
    12.0: 12,
    13.0: 13,
    14.0: 14,
    18: 18,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10
}
giris_kulub_mapping = {'Hayır': 0, 'Evet': 1}
spor_mapping = {'Hayır': 0, 'Evet': 1}
spor_rol_mapping = {'0': 0, '-': 0, 'Diğer': 0, 'DİĞER': 0, 'Bireysel': 1, 'Takım Oyuncusu': 1, 'Lider/Kaptan': 2,
                    'KAPTAN / LİDER': 2, 'Kaptan': 2}
stk_mapping = {'Hayır': 0, 'Evet': 1}
stk2_mapping = {'Hayır': 0, 'Evet': 1}
girisim_mapping = {'Hayır': 0, 'Evet': 1}
ing_mapping = {'Hayır': 0, 'Evet': 1}
ing_seviye_mapping = {'0': 0, 'Başlangıç': 1, 'Orta': 2, 'İleri': 3}

mappings = {
    'Universite_Kacinci_Sinif': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 'Yüksek Lisans': 7, 'Mezun': 0},
    'Burs_Aliyor_mu': {'Evet': 1, 'Hayır': 0},
    'Cinsiyet': {'Erkek': 1, 'Kadın': 2, 'Belirtmek istemiyorum': 0},
    'Universite_Turu': {'Devlet': 1, 'Özel': 0},
    'Universite_Not_Ortalamasi': {'0 - 1.79': 0, '1.80 - 2.49': 1, '2.50 - 3.00': 2, '3.00 - 3.49': 3, '3.50 - 4.00': 4}
}

for col, mapping in mappings.items():
    train[col] = train[col].map(mapping)

# Drop unnecessary columns
columns_to_drop = ['Lise_Adi', 'Lise_Adi_Diger', 'Lise_Bolum_Diger', 'Dogum_Tarihi', 'Burs_Miktari']
train.drop(columns=columns_to_drop, inplace=True)

# Clean column names
train.columns = [clean_column_name(col) for col in train.columns]

# Label Encoding
label_encoders = {}
for column in ['Dogum_Yeri_Temizlenmiş', 'Ikametgah_Sehri_Temizlenmiş', 'Lise_Sehir', 'Cinsiyet', 'Universite_Adi']:
    le = LabelEncoder()
    train[column] = le.fit_transform(train[column].astype(str))
    label_encoders[column] = le

# Standard Scaling
scaler = StandardScaler()
columns_to_scale = ['Dogum_Yeri_Temizlenmiş', 'Ikametgah_Sehri_Temizlenmiş', 'Lise_Sehir', 'Cinsiyet', 'Universite_Adi']
train[columns_to_scale] = scaler.fit_transform(train[columns_to_scale])

# Fill remaining NaNs with 0
train = train.fillna(0)

# Prepare data for XGBoost
X = train.drop(columns=['Degerlendirme_Puani'])
y = train['Degerlendirme_Puani']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse}")
