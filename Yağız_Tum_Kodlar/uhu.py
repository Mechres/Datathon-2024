import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Verileri yükleme
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_x.csv')

# Doğum Tarihi sütununu kaldırma
train_df = train_df.drop(columns=['Dogum Tarihi'])
test_df = test_df.drop(columns=['Dogum Tarihi'])

# Özellikler ve hedef değişken
features = [
    'Basvuru Yili', 'Cinsiyet', 'Dogum Yeri', 'Ikametgah Sehri',
    'Universite Adi', 'Universite Turu', 'Burslu ise Burs Yuzdesi', 'Burs Aliyor mu?',
    'Bölüm', 'Universite Kacinci Sinif', 'Universite Not Ortalamasi',
    'Daha Once Baska Bir Universiteden Mezun Olmus', 'Lise Adi', 'Lise Adi Diger',
    'Lise Sehir', 'Lise Turu', 'Lise Bolumu', 'Lise Bolum Diger', 'Lise Mezuniyet Notu',
    'Baska Bir Kurumdan Burs Aliyor mu?', 'Burs Aldigi Baska Kurum',
    'Baska Kurumdan Aldigi Burs Miktari', 'Anne Egitim Durumu', 'Anne Calisma Durumu',
    'Anne Sektor', 'Baba Egitim Durumu', 'Baba Calisma Durumu', 'Baba Sektor',
    'Kardes Sayisi', 'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
    'Uye Oldugunuz Kulubun Ismi', 'Profesyonel Bir Spor Daliyla Mesgul musunuz?',
    'Spor Dalindaki Rolunuz Nedir?', 'Aktif olarak bir STK üyesi misiniz?',
    'Hangi STK\'nin Uyesisiniz?', 'Stk Projesine Katildiniz Mi?',
    'Girisimcilikle Ilgili Deneyiminiz Var Mi?', 'Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?',
    'Ingilizce Biliyor musunuz?', 'Ingilizce Seviyeniz?', 'Daha Önceden Mezun Olunduysa, Mezun Olunan Üniversite'
]

# Özellikleri ve hedef değişkeni ayırma
X_train = train_df[features]
y_train = train_df['Degerlendirme Puani']
X_test = test_df[features]

# Kategorik değişkenleri sayısal değerlere dönüştürme
label_encoders = {}
for feature in features:
    if X_train[feature].dtype == 'object':
        le = LabelEncoder()
        all_labels = np.concatenate([X_train[feature].astype(str).unique(), X_test[feature].astype(str).unique()])
        le.fit(all_labels)
        X_train[feature] = le.transform(X_train[feature].astype(str))
        X_test[feature] = le.transform(X_test[feature].astype(str))
        label_encoders[feature] = le

# Eksik verileri yönetme
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# NaN ve sonsuz değerleri yönetme
y_train = y_train.replace([np.inf, -np.inf], np.nan)
y_train = y_train.fillna(y_train.median())

# Modeli oluşturma ve eğitme
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(X_train, y_train)

# Test verisi ile tahmin yapma
y_pred = model.predict(X_test)

# Gerçek hedef değerler mevcutsa RMSE hesaplama
# y_test = ... # Gerçek test hedef değerleri buraya eklenmeli
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"RMSE: {rmse}")

# Sonuçları kaydetme
submission = pd.DataFrame({
    'id': test_df['id'],
    'Degerlendirme Puani': y_pred
})
submission.to_csv('submission.csv', index=False)

print("Tahminler başarıyla kaydedildi!")
