import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# veri yukleme
data = pd.read_csv('train.csv')
print(data.describe())

print(data.isnull().sum())


data = data.dropna(subset=['Basvuru Yili'])
data = data.dropna(subset=['Degerlendirme Puani'])
data = data.dropna(subset=['Cinsiyet'])
data = data.dropna(subset=['Dogum Tarihi'])
data['Dogum Yeri'] = data['Dogum Yeri'].fillna('Bilinmiyor')
data['Ikametgah Sehri'] = data['Ikametgah Sehri'].fillna('Bilinmiyor')
data['Universite Adi'] = data['Universite Adi'].fillna('Bilinmiyor')
data['Universite Turu'] = data['Universite Turu'].fillna('Bilinmiyor')
data['Burslu ise Burs Yuzdesi'] = data['Burslu ise Burs Yuzdesi'].fillna(data['Burslu ise Burs Yuzdesi'].mean())
data['Burs Aliyor mu?'] = data['Burs Aliyor mu?'].fillna('Bilinmiyor')
data['Bölüm'] = data['Bölüm'].fillna('Bilinmiyor')
#data['Universite Kacinci Sinif'] = data['Universite Kacinci Sinif'].fillna(data['Universite Kacinci Sinif'].mean()) # Tüm row'u silmek daha iyi olabilir, deneyelim. Veri tamamen numerik değil.
data = data.dropna(subset=['Universite Kacinci Sinif'])
data['Universite Not Ortalamasi'] = data['Universite Not Ortalamasi'].fillna('0')

data['Daha Once Baska Bir Universiteden Mezun Olmus'] = data['Daha Once Baska Bir Universiteden Mezun Olmus'].fillna(
    'Bilinmiyor')

data['Lise Adi'] = data['Lise Adi'].fillna('Bilinmiyor')
data['Lise Adi Diger'] = data['Lise Adi Diger'].fillna('Bilinmiyor')
data['Lise Sehir'] = data['Lise Sehir'].fillna('Bilinmiyor')

#data[''] = data[''].fillna('Bilinmiyor')

data['Lise Turu'] = data['Lise Turu'].fillna('Bilinmiyor')
data['Lise Bolumu'] = data['Lise Bolumu'].fillna('Bilinmiyor')
data['Lise Bolum Diger'] = data['Lise Bolum Diger'].fillna('Bilinmiyor')
data['Lise Mezuniyet Notu'] = data['Lise Mezuniyet Notu'].fillna('Bilinmiyor')
data['Baska Bir Kurumdan Burs Aliyor mu?'] = data['Baska Bir Kurumdan Burs Aliyor mu?'].fillna('Bilinmiyor')
data['Burs Aldigi Baska Kurum'] = data['Burs Aldigi Baska Kurum'].fillna('Bilinmiyor')
data['Baska Kurumdan Aldigi Burs Miktari'] = data['Baska Kurumdan Aldigi Burs Miktari'].fillna('Bilinmiyor')

data['Anne Egitim Durumu'] = data['Anne Egitim Durumu'].fillna('Bilinmiyor')
data['Anne Calisma Durumu'] = data['Anne Calisma Durumu'].fillna('Bilinmiyor')
data['Anne Sektor'] = data['Anne Sektor'].fillna('0')
data['Baba Egitim Durumu'] = data['Baba Egitim Durumu'].fillna('Bilinmiyor')
data['Baba Calisma Durumu'] = data['Baba Calisma Durumu'].fillna('Bilinmiyor')
data['Baba Sektor'] = data['Baba Sektor'].fillna('0')
data['Kardes Sayisi'] = data['Kardes Sayisi'].fillna(0)
# Boş bırakılmışsa Hayır olabilir.
data['Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'] = data[
    'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?'].fillna('Hayır')
data['Uye Oldugunuz Kulubun Ismi'] = data['Uye Oldugunuz Kulubun Ismi'].fillna('0')

data['Profesyonel Bir Spor Daliyla Mesgul musunuz?'] = data['Profesyonel Bir Spor Daliyla Mesgul musunuz?'].fillna(
    'Hayır')

data['Spor Dalindaki Rolunuz Nedir?'] = data['Spor Dalindaki Rolunuz Nedir?'].fillna('0')

data['Aktif olarak bir STK üyesi misiniz?'] = data['Aktif olarak bir STK üyesi misiniz?'].fillna('Hayır')

data['Hangi STK\'nin Uyesisiniz?'] = data['Hangi STK\'nin Uyesisiniz?'].fillna('0')

data['Stk Projesine Katildiniz Mi?'] = data['Stk Projesine Katildiniz Mi?'].fillna('Hayır')

data['Girisimcilikle Ilgili Deneyiminiz Var Mi?'] = data['Girisimcilikle Ilgili Deneyiminiz Var Mi?'].fillna('Hayır')
data['Ingilizce Biliyor musunuz?'] = data['Ingilizce Biliyor musunuz?'].fillna('Hayır')
data['Ingilizce Seviyeniz?'] = data['Ingilizce Seviyeniz?'].fillna('0')

print(data.isnull().sum())

ordinal_sutunlar = ['Burslu ise Burs Yuzdesi']
nominal_sutunlar = ['Cinsiyet', 'Universite Turu', 'Burs Aliyor mu?', 'Daha Once Baska Bir Universiteden Mezun Olmus',
                    'Lise Turu', 'Lise Mezuniyet Notu', 'Baska Bir Kurumdan Burs Aliyor mu?', 'Anne Egitim Durumu',
                    'Anne Calisma Durumu', 'Anne Sektor', 'Baba Egitim Durumu', 'Baba Calisma Durumu', 'Baba Sektor',
                    'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
                    'Profesyonel Bir Spor Daliyla Mesgul musunuz?', 'Stk Projesine Katildiniz Mi?',
                    'Girisimcilikle Ilgili Deneyiminiz Var Mi?', 'Ingilizce Biliyor musunuz?']

categorical_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in ordinal_sutunlar:
    data[col] = le.fit_transform(data[col].astype(str))
for col in data.columns:
    data[col] = data[col].astype(float)

data = pd.get_dummies(data, columns=nominal_sutunlar, drop_first=True)



data = data.drop('Dogum Tarihi', axis=1)

print(data.head())

scaler = StandardScaler()

train_data = data


X = train_data.drop(['Degerlendirme Puani'], axis=1)
y = train_data['Degerlendirme Puani']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on validation set
val_predictions = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, val_predictions)
r2 = r2_score(y_val, val_predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")