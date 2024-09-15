import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('train.csv')

# Drop rows with missing values in specific columns
data = data.dropna(subset=['Basvuru Yili', 'Degerlendirme Puani', 'Cinsiyet', 'Dogum Tarihi'])

# Fill missing values with 'Bilinmiyor'
data[['Dogum Yeri', 'Ikametgah Sehri', 'Universite Adi', 'Universite Turu', 'Burslu ise Burs Yuzdesi',
      'Burs Aliyor mu?', 'Bölüm', 'Lise Adi', 'Lise Adi Diger', 'Lise Sehir',
      'Lise Turu', 'Lise Mezuniyet Notu', 'Baska Bir Kurumdan Burs Aliyor mu?',
      'Burs Aldigi Baska Kurum', 'Baska Kurumdan Aldigi Burs Miktari',
      'Anne Egitim Durumu', 'Anne Calisma Durumu', 'Anne Sektor',
      'Baba Egitim Durumu', 'Baba Calisma Durumu', 'Baba Sektor',
      'Kardes Sayisi', 'Uye Oldugunuz Kulubun Ismi', 'Hangi STK\'nin Uyesisiniz?']] = \
    data[['Dogum Yeri', 'Ikametgah Sehri', 'Universite Adi', 'Universite Turu', 'Burslu ise Burs Yuzdesi',
          'Burs Aliyor mu?', 'Bölüm', 'Lise Adi', 'Lise Adi Diger', 'Lise Sehir',
          'Lise Turu', 'Lise Mezuniyet Notu', 'Baska Bir Kurumdan Burs Aliyor mu?',
          'Burs Aldigi Baska Kurum', 'Baska Kurumdan Aldigi Burs Miktari',
          'Anne Egitim Durumu', 'Anne Calisma Durumu', 'Anne Sektor',
          'Baba Egitim Durumu', 'Baba Calisma Durumu', 'Baba Sektor',
          'Kardes Sayisi', 'Uye Oldugunuz Kulubun Ismi', 'Hangi STK\'nin Uyesisiniz?']].fillna('Bilinmiyor')

# Fill missing values with mean for numerical columns
data['Universite Kacinci Sinif'] = data['Universite Kacinci Sinif'].fillna(0)
data['Universite Not Ortalamasi'] = data['Universite Not Ortalamasi'].fillna(0)

def parse_date(date_str):
    date_formats = ['%d.%m.%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    return pd.NaT

# Parse date columns
date_cols = ['Dogum Tarihi']
for col in date_cols:
    data[col] = data[col].apply(parse_date)


# Drop date column with missing values
data = data.drop('Dogum Tarihi', axis=1)

# One-hot encode nominal columns
nominal_cols = ['Cinsiyet', 'Universite Turu', 'Burs Aliyor mu?', 'Daha Once Baska Bir Universiteden Mezun Olmus',
                'Lise Turu', 'Lise Mezuniyet Notu', 'Baska Bir Kurumdan Burs Aliyor mu?', 'Anne Egitim Durumu',
                'Anne Calisma Durumu', 'Anne Sektor', 'Baba Egitim Durumu', 'Baba Calisma Durumu', 'Baba Sektor',
                'Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?',
                'Profesyonel Bir Spor Daliyla Mesgul musunuz?', 'Stk Projesine Katildiniz Mi?',
                'Girisimcilikle Ilgili Deneyiminiz Var Mi?', 'Ingilizce Biliyor musunuz?']

le = LabelEncoder()
for col in nominal_cols:
    data[col] = le.fit_transform(data[col])

ohe = OneHotEncoder(sparse_output=False)
encoded_data = ohe.fit_transform(data[nominal_cols])
encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(nominal_cols))

data = pd.concat([data, encoded_df], axis=1)

# Scale numerical columns
scaler = StandardScaler()
numerical_cols = [col for col in data.columns if data[col].dtype.kind not in ('b', 'c')]
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Save to Excel
data.to_excel('test2.xlsx')
