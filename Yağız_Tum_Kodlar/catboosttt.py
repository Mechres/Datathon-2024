import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Verileri yükleyelim
train_data = pd.read_csv('train.csv', low_memory=False)
test_data = pd.read_csv('test_x.csv', low_memory=False)
submission = pd.read_csv('sample_submission.csv')

# Sütunları kontrol edelim
print("Train columns:", train_data.columns)
print("Test columns:", test_data.columns)

# Tarih formatındaki sütunları datetime tipine dönüştürelim
train_data['Dogum Tarihi'] = pd.to_datetime(train_data['Dogum Tarihi'], errors='coerce', dayfirst=True)
test_data['Dogum Tarihi'] = pd.to_datetime(test_data['Dogum Tarihi'], errors='coerce', dayfirst=True)

# Tarih sütunlarını işleyelim
train_data['Dogum Tarihi'] = train_data['Dogum Tarihi'].fillna(pd.Timestamp('1900-01-01'))
test_data['Dogum Tarihi'] = test_data['Dogum Tarihi'].fillna(pd.Timestamp('1900-01-01'))

# Kategorik sütunları belirleyelim
categorical_columns = ['Cinsiyet', 'Dogum Yeri', 'Ikametgah Sehri', 'Universite Adi', 'Universite Turu',
                       'Burs Aliyor mu?', 'Bölüm', 'Universite Kacinci Sinif', 'Universite Not Ortalamasi',
                       'Daha Once Baska Bir Universiteden Mezun Olmus', 'Lise Adi', 'Lise Adi Diger', 'Lise Turu', 'Lise Bolumu',
                       'Lise Sehir', 'Lise Bolum Diger', 'Lise Mezuniyet Notu',
                       'Baska Bir Kurumdan Burs Aliyor mu?','Burs Aldigi Baska Kurum', 'Baska Kurumdan Aldigi Burs Miktari', 'Anne Egitim Durumu', 'Anne Calisma Durumu',
                       'Anne Sektor', 'Baba Egitim Durumu', 'Baba Calisma Durumu', 'Baba Sektor',
                       'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?', 'Uye Oldugunuz Kulubun Ismi', 'Profesyonel Bir Spor Daliyla Mesgul musunuz?',
                       'Spor Dalindaki Rolunuz Nedir?',  'Hangi STK\'nin Uyesisiniz?',
                       'Aktif olarak bir STK üyesi misiniz?', 'Stk Projesine Katildiniz Mi?', 'Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?',
                       'Girisimcilikle Ilgili Deneyiminiz Var Mi?', 'Ingilizce Biliyor musunuz?', 'Ingilizce Seviyeniz?', 'Daha Önceden Mezun Olunduysa, Mezun Olunan Üniversite']

# Tekrar eden sütun adlarını kaldırma
categorical_columns = list(set(categorical_columns))

# Eksik sütunları kontrol etme
missing_columns = [col for col in categorical_columns if col not in train_data.columns]
print("Missing columns:", missing_columns)

# Eksik sütunları çıkarmak
categorical_columns = [col for col in categorical_columns if col in train_data.columns]

# Kategorik sütunları eksik veriler için dolduralım
train_data[categorical_columns] = train_data[categorical_columns].fillna('Bilinmiyor')
test_data[categorical_columns] = test_data[categorical_columns].fillna('Bilinmiyor')



# Eğitim ve test setindeki sütunları uyumlu hale getirelim
common_columns = [col for col in train_data.columns if col in test_data.columns]

# 'Degerlendirme Puani' ve 'id' sütunları varsa düşürelim
target_column = 'Degerlendirme Puani'
id_column = 'id'

if target_column in common_columns:
    common_columns.remove(target_column)
if id_column in common_columns:
    common_columns.remove(id_column)

X = train_data[common_columns]
y = train_data[target_column] if target_column in train_data.columns else None
test_X = test_data[common_columns].drop(id_column, axis=1, errors='ignore')  # 'id' sütununu düşürme, eğer varsa

# Kategorik sütunları belirleyelim
categorical_features = [i for i, col in enumerate(X.columns) if col in categorical_columns]

# Verileri eğitim ve doğrulama setlerine bölelim
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost modelini oluşturalım ve eğitelim
model = CatBoostRegressor(cat_features=categorical_features, verbose=100, random_seed=42)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# Modeli değerlendirelim
y_pred = model.predict(X_val)
print('Validation RMSE:', mean_squared_error(y_val, y_pred, squared=False))

# Test seti için tahmin yapalım
test_predictions = model.predict(test_X)

# Tahminleri submission formatına uygun hale getirelim
submission['Degerlendirme Puani'] = test_predictions

# Sonuçları kaydedelim
submission.to_csv('submission.csv', index=False)
