import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler,RobustScaler
from sklearn.impute import SimpleImputer
import os

# =====================================================================
# 1. Adım: TRAIN / TEST SPLIT (Feyza'nın Önerisi - Data Leakage Önlemi)
# =====================================================================
y_train = None
y_test = None

def split_train_test(df):
    global y_train, y_test
    y = df[['treatment', 'turnover_risk']]
    X = df.drop(columns=['treatment','turnover_risk'])

    df_mental_health_clean_target = df[['treatment', 'turnover_risk']].copy()  # Yedekleme, gerekirse kullanmak için

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test 


# =======================================================================================================
#  2. Adım: EN ÇOK TEKRAR EDEN DEĞERLE (MOD) DOLDURMA, AYKIRI DEĞERLERİ DÜZELTME, BOŞ VERİLERİ DOLDURMA
# ========================================================================================================
def fill_missing_values(df_xtrain, df_xtest):
    """
    Bu fonksiyon, eksik değerleri doldurmak için farklı stratejiler uygular.
    - comments sütunu için "No Comment" ile doldurma
    - state sütunu için "Unknown" ile doldurma
    - Kategorik sütunlar için mod (en çok tekrar eden değer) ile doldurma
    """

    # comments sütununa boş bırakanlar için "No Comment" yazıyoruz
    df_xtrain['comments'] = df_xtrain['comments'].fillna("N/A")
    df_xtest['comments'] = df_xtest['comments'].fillna("N/A")

    #### Aykırı yaş değerlerini düzenliyoruz ##########
    temiz_yaslar_xtrain = df_xtrain[(df_xtrain['Age'] >= 18) & (df_xtrain['Age'] <= 75)]
    yas_medyani_xtrain = temiz_yaslar_xtrain['Age'].median()
    # 3. Şimdi 18'den küçük veya 75'ten büyük olan tüm o 19 saçma değeri yakala
    # Ve hepsinin içine bulduğumuz o mantıklı medyan değerini (Örn: 32) yapıştır.
    df_xtrain.loc[(df_xtrain['Age'] < 18) | (df_xtrain['Age'] > 75), 'Age'] = yas_medyani_xtrain
    df_xtrain['Age'] = df_xtrain['Age'].fillna(yas_medyani_xtrain)

    temiz_yaslar_xtest = df_xtest[(df_xtest['Age'] >= 18) & (df_xtest['Age'] <= 75)]
    yas_medyani_xtest = temiz_yaslar_xtest['Age'].median()
    df_xtest.loc[(df_xtest['Age'] < 18) | (df_xtest['Age'] > 75), 'Age'] = yas_medyani_xtest
    df_xtest['Age'] = df_xtest['Age'].fillna(yas_medyani_xtrain)


    print(f"Aykırı değerlerin yerine atanan medyan yaş - Train: {yas_medyani_xtrain}")
    print(f"Aykırı değerlerin yerine atanan medyan yaş - Test: {yas_medyani_xtest}")

    # state sütununa boş bırakanlar (ABD dışındakiler) için "Unknown" yazıyoruz
    df_xtrain['state'] = df_xtrain['state'].fillna("Unknown")
    df_xtest['state'] = df_xtest['state'].fillna("Unknown")


    # Şirket içi kategorik sütunların listesi
    sirket_kolonlari = [
            'self_employed',
            'family_history',
            'mental_health_consequence',
            'phys_health_consequence',
            'coworkers',
            'supervisor',
            'mental_health_interview',
            'phys_health_interview',
            'mental_vs_physical',
            'obs_consequence',
            'work_interfere',
            'no_employees',
            'remote_work',
            'tech_company',
            'benefits',
            'care_options',
            'wellness_program',
            'seek_help',
            'anonymity',
            'leave'
    ]

    # Hepsini tek tek yazmak yerine bir döngüyle en çok tekrar eden değerlerine eşitleyelim
    for kolon in sirket_kolonlari:
        en_cok_tekrar_eden_xtrain  = df_xtrain[kolon].mode()[0] # En sık girilen cevabı bulur
        df_xtrain[kolon] = df_xtrain[kolon].fillna(en_cok_tekrar_eden_xtrain)
        en_cok_tekrar_eden_xtest = df_xtest[kolon].mode()[0] # En sık girilen cevabı bulur
        df_xtest[kolon] = df_xtest[kolon].fillna(en_cok_tekrar_eden_xtest)

    # Kişisel bilgiler (Gender ve Country) için mod doldurması
    df_xtrain['Gender'] = df_xtrain['Gender'].fillna("Other/Prefer Not To Say")
    df_xtest['Gender'] = df_xtest['Gender'].fillna("Other/Prefer Not To Say")

    df_xtrain['Country'] = df_xtrain['Country'].fillna(df_xtrain['Country'].mode()[0])
    df_xtest['Country'] = df_xtest['Country'].fillna(df_xtest['Country'].mode()[0])

    # KONTROL: Boş veri kaldı mı diye bakıyoruz
    print("Kalan eksik veri sayıları:")
    print(f"Train verisindeki boş verilerin sayısı: {df_xtrain.isnull().sum().sum()}")
    print(f"Test verisindeki boş verilerin sayısı: {df_xtest.isnull().sum().sum()}")

    return df_xtrain, df_xtest

# =====================================================================
# 3. Adım: FEATURE EXTRACTION (Yeni Özellikleri Ham Veriden Türetme)
# =====================================================================
def features_extract(df_xtrain,df_xtest):
    df_new_xtrain = df_xtrain.copy()
    df_new_xtest = df_xtest.copy()

    # --- A) Kurumsal Destek Skoru (0-5 Arası) ---
    support_cols = ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity']
    # Her 'Yes' cevabı 1 puan, diğerleri 0 puan
    df_new_xtrain['corporate_support_score'] = df_new_xtrain[support_cols].apply(
          lambda row: sum(row.isin(['Yes', 1, '1'])), axis=1)    
    df_new_xtest['corporate_support_score'] = df_new_xtest[support_cols].apply(          
        lambda row: sum(row.isin(['Yes', 1, '1'])), axis=1) 

    # --- B) Psikolojik Güvenli Alan Skoru (0-3 Arası) ---
    safety_cols = ['supervisor', 'coworkers', 'anonymity']
    # Her 'Yes' cevabı 1 puan, diğerleri 0 puan
    df_new_xtrain['psychological_safety_score'] = df_new_xtrain[safety_cols].apply(
        lambda row: sum(row.isin(['Yes', 1, '1'])), axis=1)
    df_new_xtest['psychological_safety_score'] = df_new_xtest[safety_cols].apply(
        lambda row: sum(row.isin(['Yes', 1, '1'])), axis=1)
    
    # --- C) YENİ ÖNERİ 1: Damgalanma Endişesi Skoru (Stigma Score) (0-3 Arası) ---
    # İşyerinde akıl sağlığı konusunun negatif algılanma derecesi
    df_new_xtrain['stigma_score'] = (
        (df_new_xtrain['mental_health_consequence'].isin(['Yes', 1, '1'])).astype(int) +
        (df_new_xtrain['obs_consequence'].isin(['Yes', 1, '1'])).astype(int) +
        (df_new_xtrain['mental_vs_physical'].isin(['No', 0, '0'])).astype(int) # Fiziksel kadar ciddi görülmüyorsa +1 risk
    )

    df_new_xtest['stigma_score'] = (
        (df_new_xtest['mental_health_consequence'].isin(['Yes', 1, '1'])).astype(int) +
        (df_new_xtest['obs_consequence'].isin(['Yes', 1, '1'])).astype(int) +
        (df_new_xtest['mental_vs_physical'].isin(['No', 0, '0'])).astype(int) # Fiziksel kadar ciddi görülmüyorsa +1 risk
    )
    
    # --- D) YENİ ÖNERİ 2: İletişim Bariyeri İndeksi (0 veya 1) ---
    # Kişi yöneticisiyle fiziksel sağlığı konuşabiliyor ama mental sağlığı konuşamıyorsa bariyer vardır
    # OSMI veri setinde 'supervisor' mental, eğer verinde 'phys_supervisor' varsa ikisini kıyaslayabilirsin.
    # Yoksa doğrudan yöneticiden çekinme durumunu binary yapalım:
    df_new_xtrain['has_communication_barrier'] = df_new_xtrain['supervisor'].isin(['No', 'Don\'t know']).astype(int)
    df_new_xtest['has_communication_barrier'] = df_new_xtest['supervisor'].isin(['No', 'Don\'t know']).astype(int)

    return df_new_xtrain, df_new_xtest

# =====================================================================
# 4. Adım: FEATURE DROP (İşe yaramayacak özellikleri siliyoruz)
# =====================================================================
def features_drop(df_xtrain,df_xtest):
    silinecek_sutunlar = [
        'comments',                  # Pas geçtiğimiz serbest metin
        'benefits', 'care_options',  # Bunları zaten corporate_support_score içinde topladık
        'wellness_program', 'seek_help', 'anonymity',
        'mental_health_consequence', 'obs_consequence', 'mental_vs_physical',
        'Timestamp', 'UserID', 'SurveyID'
    ]
    
    # Hem train hem testten bu sütunları tamamen kaldırıyoruz
    X_train_final = df_xtrain.drop(columns=silinecek_sutunlar, errors='ignore')
    X_test_final = df_xtest.drop(columns=silinecek_sutunlar, errors='ignore')

    print("🚀 Veri hazırlığı bitti! Sütunlar düşürüldü, modellemeye hazırız.")
    
    return X_train_final, X_test_final

# =====================================================================
# 5. Adım: ENCODING & SCALING (İşe yaramayacak özellikleri siliyoruz)
# =====================================================================

def apply_encoding_and_scaling(X_train, X_test):
    """
    3. Hafta analizleri bittikten sonra, 4. hafta ML modellerine (XGBoost/RF) 
    girmeden hemen önce çalıştırılacak nihai dönüşüm fonksiyonudur.
    """
    # Kopya üzerinden çalışalım ki pandas amca uyarı fırlatmasın
    X_train_final = X_train.copy()
    X_test_final = X_test.copy()
    
    work_interfere_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
    leave_map = {
        'Very easy': 0, 'Somewhat easy': 1, 
        'Don\'t know': 2, 'Neither easy nor difficult': 2, 'I don\'t know': 2,
        'Somewhat difficult': 3, 'Very difficult': 4, 'Difficult': 4
    }

    no_employees_map = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5}
    
    for df_encoded in [X_train_final, X_test_final]:
        df_encoded['work_interfere'] = df_encoded['work_interfere'].map(work_interfere_map)
        df_encoded['leave'] = df_encoded['leave'].map(leave_map)
        df_encoded['no_employees'] = df_encoded['no_employees'].map(no_employees_map)

    # =============ONE-HOT ENCODING (Nominal Sütunlar İçin)=======================
    # 'Country', 'state' Target Encoding işlemine tabi tutulacağı için çıkarıldı
    nominal_cols = [
        'Gender',  'self_employed', 'family_history', 
        'remote_work', 'tech_company', 'coworkers', 'supervisor', 
        'mental_health_interview', 'phys_health_interview', 'phys_health_consequence'
    ]
    
    # OneHotEncoder nesnesini oluşturuyoruz
    # handle_unknown='ignore' kullanırken drop_first=True sorun yaratmasın diye sparse_output=False yapıyoruz
    ohe = OneHotEncoder(drop='first', sparse_output=False)    # Kuralı SADECE train üzerinden öğreniyoruz (Fit)
    ohe.fit(X_train_final[nominal_cols])
    
    # Train ve Test verilerini dönüştürüyoruz (Transform)
    train_ohe_array = ohe.transform(X_train_final[nominal_cols])
    test_ohe_array = ohe.transform(X_test_final[nominal_cols])
    
    # Yeni oluşan 1-0 sütunlarının isimlerini alıyoruz
    ohe_columns = ohe.get_feature_names_out(nominal_cols)
    
    # Array'leri DataFrame formatına getiriyoruz
    train_ohe_df = pd.DataFrame(train_ohe_array, columns=ohe_columns, index=X_train_final.index)
    test_ohe_df = pd.DataFrame(test_ohe_array, columns=ohe_columns, index=X_test_final.index)
    
    # Eski ham nominal sütunları düşürüp, yeni one-hot sütunlarını yana ekliyoruz (Concat)
    X_train_final = X_train_final.drop(columns=nominal_cols)
    X_train_final = pd.concat([X_train_final, train_ohe_df], axis=1)
    
    X_test_final = X_test_final.drop(columns=nominal_cols)
    X_test_final = pd.concat([X_test_final, test_ohe_df], axis=1)

    # =============Country ve State için TARGET ENCODING uygulanıyor =======================
    # Hedeflerimiz: 'treatment' ve 'turnover_risk'
    # Kodlayacağımız kolonlar: 'Country' ve 'state'

    # ==========================================
    # 1. 'Country' Kolonunu 2 Target İçin Kodlama
    # ==========================================
    # Train üzerinden haritaları öğreniyoruz
    c_treat_map, c_treat_mean = target_encode_train(X_train_final, 'Country', y_train['treatment'])
    c_turn_map, c_turn_mean = target_encode_train(X_train_final, 'Country', y_train['turnover_risk'])
    # Hem Train hem Test setine bu oranları basıyoruz
    
    X_train_final = target_encode_transform(X_train_final, 'Country', 'treatment', c_treat_map, c_treat_mean)
    X_train_final = target_encode_transform(X_train_final, 'Country', 'turnover_risk', c_turn_map, c_turn_mean)
    X_test_final = target_encode_transform(X_test_final, 'Country', 'treatment', c_treat_map, c_treat_mean)
    X_test_final = target_encode_transform(X_test_final, 'Country', 'turnover_risk', c_turn_map, c_turn_mean)

    # ==========================================
    # 2. 'state' Kolonunu 2 Target İçin Kodlama
    # ==========================================
    # Train üzerinden haritaları öğreniyoruz
    s_treat_map, s_treat_mean = target_encode_train(X_train_final, 'state', y_train['treatment'])
    s_turn_map, s_turn_mean = target_encode_train(X_train_final, 'state', y_train['turnover_risk'])

    # Hem Train hem Test setine bu oranları basıyoruz
    X_train_final = target_encode_transform(X_train_final, 'state', 'treatment', s_treat_map, s_treat_mean)
    X_train_final = target_encode_transform(X_train_final, 'state', 'turnover_risk', s_turn_map, s_turn_mean)
    X_test_final = target_encode_transform(X_test_final, 'state', 'treatment', s_treat_map, s_treat_mean)
    X_test_final = target_encode_transform(X_test_final, 'state', 'turnover_risk', s_turn_map, s_turn_mean)

    # ==========================================
    # 3. Eski Metin Kolonlarını Silme (Opsiyonel ama önerilir)
    # ==========================================
    # İşimiz bittiği için orijinal metin kolonlarını atabiliriz, yerlerine nümerik versiyonları geldi
    X_train_final = X_train_final.drop(columns=['Country', 'state'])
    X_test_final = X_test_final.drop(columns=['Country', 'state'])

    #==============ROBUST SCALER (Aykırı Değerlerden Etkilenmeyen Ölçekleme)========================
    # RobustScaler nesnesini oluşturuyoruz
    scaler = RobustScaler()
    
    # Yaş sütununun kuralını SADECE train üzerinden öğreniyoruz (Fit)
    # Çift köşeli parantez [[]] DataFrame formatında kalmasını sağlar, hata önler
    scaler.fit(X_train_final[['Age']])
    
    # Train ve Test üzerindeki yaş değerlerini dönüştürüyoruz (Transform)
    X_train_final['Age'] = scaler.transform(X_train_final[['Age']])
    X_test_final['Age'] = scaler.transform(X_test_final[['Age']])

    ############ TRAIN ve TEST dosyalarını oluşturma ################
    
    df_mental_health_train = pd.concat([X_train_final, y_train], axis=1)
    df_mental_health_test = pd.concat([X_test_final, y_test], axis=1)

    df_mental_health_train.to_csv("dataset/mental_health_train.csv", index=False)    
    df_mental_health_test.to_csv("dataset/mental_health_test.csv", index=False)

    print("🚀 Tüm işlemler (Label, One-Hot ve Target Encoding & Robust Scaling) adım adım ve sıfır sızıntıyla tamamlandı!")
    return X_train_final, X_test_final

def target_encode_train(df, kolon, target_serisi):
    """
    df: X_train tablosu
    kolon: 'Country' veya 'state'
    target_serisi: y_train['treatment'] veya y_train['turnover_risk']
    """
    # X_train'deki kolon ile dışarıdaki y_train serisini eşleştirip ortalamasını alıyoruz
    oran_haritasi = target_serisi.groupby(df[kolon]).mean().to_dict()
    genel_ortalama = target_serisi.mean()
    
    return oran_haritasi, genel_ortalama
    
    return oran_haritasi, genel_ortalama

def target_encode_transform(df, kolon, target_kolonu, oran_haritasi, genel_ortalama):
    """
    Hesaplanan oranları kullanarak kolonun içindeki isimleri sayılara dönüştürür.
    """
    yeni_kolon_adi = f"{kolon}_encoded_{target_kolonu}"
    
    # İsimleri oranlarla değiştir, listede yoksa genel ortalamayı bas
    df[yeni_kolon_adi] = df[kolon].map(oran_haritasi).fillna(genel_ortalama)
    return df

def fe_summary_pickle(df):
    PICKLE_PATH = "dataset/fe_summary_file.pkl"

    """
    Pipeline fonksiyonlarından TAMAMEN bağımsız çalışır.
    Ham veri üzerinden eksik, aykırı, eklenen ve silinen bilgileri yakalar, 
    pickle dosyasına yazar ve sade bir şekilde ekrana basar.
    """
    # 1. KONTROL: Eğer pickle zaten varsa, hiç hesaplama yapma direkt oku ve göster!
    if os.path.exists(PICKLE_PATH):
        print(f"🔄 Özet sayfası verileri yerel pickle dosyasından ({PICKLE_PATH}) okundu.")
        with open(PICKLE_PATH, 'rb') as f:
            summary = pickle.load(f)
        return summary


    print("✨ Pickle dosyası bulunamadı. Ham veri üzerinden kurallar ve sayılar çıkarılıyor...")

    summary = {
        "missing_values": [],
        "outliers": {},
        "extracted_features": [],
        "dropped_features": []
    }

    # --- A) BOŞ DEĞERLER VE METOTLARI ---
    # Kodundaki kurallara göre doldurma tiplerini haritalandırıyoruz
    sirket_kolonlari = [
            'self_employed',
            'family_history',
            'mental_health_consequence',
            'phys_health_consequence',
            'coworkers',
            'supervisor',
            'mental_health_interview',
            'phys_health_interview',
            'mental_vs_physical',
            'obs_consequence',
            'work_interfere',
            'no_employees',
            'remote_work',
            'benefits',
            'care_options',
            'wellness_program',
            'seek_help',
            'anonymity',
            'leave'
    ]
    yas_medyani = df[(df['Age'] >= 18) & (df['Age'] <= 75) & (df['Age'].notnull())]['Age'].median()

    # Her bir kolonun ham verideki boş değer sayısını alma işlemi
    for col in df.columns:
        null_count = df[col].isnull().sum()
        
        # Sadece boş değer içerenleri rapora ekleyelim
        if null_count > 0:
            if col == 'state':
                method, value = "Constant String", "Unknown"
            elif col == 'Age':
                method, value = "Median Suppression", f"{yas_medyani} (Medyan Yaş)"
            elif col == 'Gender':
                method, value = "Constant String", "Other/Prefer Not To Say"
            elif col in sirket_kolonlari or col == 'Country':
                method, value = "Mode (En Çok Tekrar Eden)", str(df[col].mode()[0])
            else:
                method, value = "Bilinmeyen Metot", "N/A"
                
            summary["missing_values"].append({
                "Sütun": col,
                "Boş_Adet": int(null_count),
                "Yöntem": method,
                "Atanan_Değer": value
            })

    # --- B) AYKIRI DEĞERLER (Age) ---
    outlier_condition = (df['Age'].notnull()) & ((df['Age'] < 18) | (df['Age'] > 75))
    outlier_count = outlier_condition.sum()
    print(f"Aykırı yaş değerleri (18-75 dışında) sayısı: {outlier_count}")
    # Kodundaki gibi 18-75 arası mantıklı yaşların medyanını bulalım
    summary["outliers"] = {
        "Sütun": "Age",
        "Aykırı_Adet": int(outlier_count),
        "Kriter": "< 18 veya > 75",
        "Yöntem": "Median Suppression",
        "Atanan_Medyan": float(yas_medyani)
    }

    # --- C) EKLENEN YENİ ÖZELLİKLER ---
    # features_extract fonksiyonundaki mantıksal formüller
    summary["extracted_features"] = [
        {"Özellik": "corporate_support_score", "Tip": "Sayısal (0-5)", "Kural": "benefits, care_options, wellness_program, seek_help, anonymity toplam 'Yes' sayısı"},
        {"Özellik": "psychological_safety_score", "Tip": "Sayısal (0-3)", "Kural": "supervisor, coworkers, anonymity toplam 'Yes' sayısı"},
        {"Özellik": "stigma_score", "Tip": "Sayısal (0-3)", "Kural": "mental_health_consequence(Yes) + obs_consequence(Yes) + mental_vs_physical(No)"},
        {"Özellik": "has_communication_barrier", "Tip": "Binary (0-1)", "Kural": "supervisor 'No' veya 'Don't know' ise 1, aksi halde 0"}
    ]

    # --- D) ÇIKARILAN ÖZELLİKLER ---
    # features_drop fonksiyonundaki birebir silme listesi
    summary["dropped_features"] = [
        {"Sütun": "comments", "Neden": "Serbest metin gürültüsü ve eksik oranının yüksekliği"},
        {"Sütun": "benefits, care_options, wellness_program, seek_help", "Neden": "Bilgi sızıntısı önlemi -> corporate_support_score'a dönüştü"},
        {"Sütun": "anonymity", "Neden": "Bilgi sızıntısı önlemi -> support ve safety skorlarına gömüldü"},
        {"Sütun": "mental_health_consequence, obs_consequence, mental_vs_physical", "Neden": "Bilgi sızıntısı önlemi -> stigma_score'a dönüştü"},
        {"Sütun": "Timestamp", "Neden": "Chi-Square bağımsızlık analizinde iki hedef değişken için de tamamen ANLAMSIZ çıktı"},
        {"Sütun": "UserID, SurveyID", "Neden": "Model için bilgi taşımayan teknik tekil anahtarlar (ID)"}
    ]

    # 2. KAYDETME: Oluşan bu temiz sözlüğü pickle dosyasına yazıyoruz
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(summary, f)
    print(f"💾 Özet bilgileri başarıyla '{PICKLE_PATH}' dosyasına kilitlendi.")
    

    # Ekrana bas
    return summary

def kalici_country_temizligi(df):
    print("5'ten az frekansa sahip ülkeler 'Other' yapılıyor...")
    
    if 'Country' not in df.columns:
        print("❌ Hata: Veri setinde 'Country' kolonu bulunamadı!")
        return

    df['Country'] = df['Country'].astype(str).str.strip()
    
    # 1. Her ülkenin veri setindeki toplam sayısını hesapla
    country_counts = df['Country'].value_counts()
    
    frequent_countries = country_counts[(country_counts > 0) & (country_counts <= 5) ].index.tolist()
    
    # 3. Eğer ülke bu listede yoksa (yani sayısı 5'ten küçük veya eşitse) 'Other' yap
    df['Country'] = df['Country'].apply(lambda x: 'Other' if x in frequent_countries else x)

    # Dosyayı orijinalinin üzerine kalıcı olarak kaydet
    SAVING_PATH = "dataset/mental_health.csv"
    df.to_csv(SAVING_PATH, index=False) 
    print(f"💾 Başarılı! Nadir ülkeler 'Other' havuzuna aktarıldı ve {SAVING_PATH} adresine kaydedildi.")

def kalici_state_temizligi(df):
    print("State kolonu dengeleniyor (Frekansı 5'ten küçük olanlar 'Other' yapılıyor)...")
    
    if 'state' not in df.columns:
        print("❌ Hata: Veri setinde 'state' kolonu bulunamadı!")
        return

    df['state'] = df['state'].astype(str).str.strip()
    
    # 1. Her eyaletin toplam frekansını hesapla
    state_counts = df['state'].value_counts()
    
    frequent_states = state_counts[(state_counts > 0) & (state_counts <= 5)].index.tolist()
    
    df['state'] = df['state'].apply(lambda x: 'Other' if x in frequent_states else x)
    # Dosyayı orijinalinin üzerine kalıcı olarak kaydet
    SAVING_PATH = "dataset/mental_health.csv"
    df.to_csv(SAVING_PATH, index=False) 
    print(f"💾 Başarılı! Eyalet frekans filtrelemesi tamamlandı ve {SAVING_PATH} adresine kaydedildi.")