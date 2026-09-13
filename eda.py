import pandas as pd
import numpy as np
import csv
import streamlit as st
import os
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats

# Pickle dosyasının kaydedileceği sabit yol
PICKLE_PATH = "dataset/hatali_ve_aykiri_raporu.pkl"
gender_check = True

def analyze_incorrect_osmi_datas(df):
    """OSMI veri setindeki boş, negatif ve aykırı değerleri bulur,

    tek bir DataFrame'de toplar ve Pickle olarak diske kaydeder.
    """
    if df is None or df.empty:
        return None

    anomaly_records = []

    # --- 1. MANTIKSAL YAŞ (AGE) ANALİZİ (Görsellerdeki Sorunu Çözen Kısım) ---
    if 'Age' in df.columns:
        # Yaş sütununu sayıya çeviriyoruz (çevrilemeyen harfleri NaN yap)
        age_series = pd.to_numeric(df['Age'], errors='coerce')

        # Kuralımız: 18'den küçük VEYA 75'den büyük olan her şey gerçek bir anomalidir!
        # Böylece -1726, -29, 5,8,11, 329, 999999999 gibi tüm çöpler yakalanır.
        outliers = df[((age_series < 18) | (age_series > 75))]  # 75 yaş üstü de mantıksal olarak aykırı kabul edelim

        for idx, row in outliers.iterrows():
            anomaly_records.append({
                "Satır İndeksi": idx,
                "Hata Türü": "Outlier",
                "Kolon Adı": "Age",
                "Mevcut Değer": row['Age']
            })
        
        negative_age = age_series < 0
        negatives = df[negative_age]
        for idx, row in negatives.iterrows():
            print(f"Negatif yaş bulundu: {idx}, Değer: {row['Age']}")
            anomaly_records.append({
                "Satır İndeksi": idx,
                "Hata Türü": "Negative",
                "Kolon Adı": "Age",
                "Mevcut Değer": row['Age']
            })

    # --- 2. BOŞ VE DİĞER NEGATİF DEĞER ANALİZİ ---
    for col in df.columns:
        # Zaten Age kolonunu yukarıda özel olarak inceledik, tekrar girmesin
        #if col == 'Age':continue
            
        # Boş (NaN) Değer Kontrolü
        null_rows = df[df[col].isna()]
        for idx, row in null_rows.iterrows():
            anomaly_records.append({
                "Satır İndeksi": idx,
                "Hata Türü": "NaN",
                "Kolon Adı": col,
                "Mevcut Değer": "Eksik Veri"
            })
        
        if col != 'Age':
        # Diğer sayısal kolonlar için negatif değer kontrolü
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            neg_numeric = df[numeric_series < 0]
            if df[col].dtype == object:  # metin sütunu
                neg_strings = df[df[col].astype(str).str.startswith('-')]  #& df[col].astype(str).str.match(r'^-\d+')]
                # bu negatif sayı içeren metinleri yakalar
            else:
                neg_strings = pd.DataFrame()  # boş

            # Birleştir
            negative_rows = pd.concat([neg_numeric, neg_strings]).drop_duplicates()
            
            if not negative_rows.empty:
                print(f"NEGATİF DEGER BULUNDU! {negative_rows.shape[0]} satır, Kolon: {col}")

            for idx, row in negative_rows.iterrows():
                print(f"Negatif değer bulundu: For döngüsü {idx}, Kolon {col}, Değer {row[col]}")
                anomaly_records.append({
                    "Satır İndeksi": idx,
                    "Hata Türü": "Negative",
                    "Kolon Adı": col,
                    "Mevcut Değer": row[col]
                })

    # --- 3. RAPORU OLUŞTURMA VE PICKLE KAYDI ---
    if anomaly_records:
        report_df = pd.DataFrame(anomaly_records)
    else:
        report_df = pd.DataFrame(columns=["Satır İndeksi", "Hata Türü", "Kolon Adı", "Mevcut Değer"])

    # Pickle'a yazıp donduruyoruz
    report_df.to_pickle(PICKLE_PATH)
    return report_df

def load_pickle_file():
    if os.path.exists(PICKLE_PATH):
        return pd.read_pickle(PICKLE_PATH)
    return None


def available_columns(df):
    """Veriyi okur, SurveyID dahil analiz dışı kolonları eler."""
    
    # Analiz edilmeyecek kolonlar eleniyor (SurveyID dahil)
    exclude_cols = ['Timestamp', 'comments', 'UserID', 'SurveyID']
    available_columns = [col for col in df.columns if col not in exclude_cols]
    
    return available_columns

def generate_cross_plot(df, x_axis_col, hue_col):
    """Sadece veri mantığını yönetir, tasarımı harici dosyadan çeker."""
    plot_df = df[[x_axis_col, hue_col]].fillna('NaN/Empty')
        
    crosstab_res = pd.crosstab(plot_df[x_axis_col], plot_df[hue_col])
    
    # Tüm görsel şablonu harici .mplstyle dosyasından yüklüyoruz
    graph_style = {
        'figure.facecolor': '#f4f7f6',  # CSS .main arka planınla birebir aynı!
        'axes.facecolor': '#ffffff',
        'axes.grid': True,
        'grid.color': '#f1f5f9',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'text.color': '#475569',
        'axes.titleweight': 'bold',
        'axes.titlesize': 13,
        'xtick.color': '#475569',
        'ytick.color': '#475569'
    }
    
    # Matplotlib'e doğrudan hafızadaki bu stili basıyoruz (Asla patlamaz)
    plt.rcParams.update(graph_style)

    
    fig, ax = plt.subplots(figsize=(15, 9))    
    # Grafiği çiziyoruz (CSS renk paletinle uyumlu Set3 paleti)
    crosstab_res.plot(
        kind='bar', 
        stacked=True, 
        ax=ax, 
        cmap='Set3', 
        edgecolor='#cbd5e1', 
        linewidth=1,
        width=0.55
    )
    
    # Dinamik metin etiketleri
    ax.set_title(f"{x_axis_col} Baskınlığında {hue_col} Dağılım Analizi", pad=18)
    ax.set_xlabel(x_axis_col, labelpad=10)
    ax.set_ylabel("Toplam Kayıt Sayısı", labelpad=10)
    
    plt.xticks(rotation=25, ha='right')
    
    # Sağ taraftaki gösterge kartı (Legend)
    ax.legend(
        title=hue_col, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.25), 
        ncol=7,                      # Yan yana 7 sütun yaparak dikey kalabalığı dağıttık
        frameon=True, 
        edgecolor='#e2e8f0',
        title_fontsize=10,
        fontsize=9                   # Okunabilirliği korumak için ideal boyut
    )
    
    # Sıkışmayı tamamen önlemek için alt payı manuel genişletiyoruz
    plt.subplots_adjust(bottom=0.35)
    
    plt.tight_layout()
    return fig, crosstab_res


def generate_cross_plotly(df, x_axis_col, hue_col):
    """
    Karanlık tema ile uyumlu, jilet gibi net okunabilen 
    interaktif Plotly grafiği.
    """
    plot_df = df[[x_axis_col, hue_col]].fillna('NaN/Empty')
    crosstab_res = pd.crosstab(plot_df[x_axis_col], plot_df[hue_col]).reset_index()
    
    melted_df = pd.melt(
        crosstab_res, 
        id_vars=[x_axis_col], 
        var_name=hue_col, 
        value_name='Toplam Kayıt Sayısı'
    )
    
    fig = px.bar(
        melted_df, 
        x=x_axis_col, 
        y='Toplam Kayıt Sayısı', 
        color=hue_col,
        title=f"<b>{x_axis_col} Baskınlığında {hue_col} Dağılım Analizi</b>",
        template='plotly_dark', # 🎯 SİHİRLİ DOKUNUŞ: Grafiği tamamen karanlık temaya geçirdik!
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Arka planı senin Streamlit panelinin o tatlı siyahı ile eşitliyoruz
    fig.update_layout(
        height=650,              # 🎯 SİHİRLİ DOKUNUŞ: Grafiğin yüksekliğini 650 piksele çıkarıyoruz!
        xaxis_title=x_axis_col,
        yaxis_title="Toplam Kayıt Sayısı",
        barmode='stack',
        paper_bgcolor='#0e1117', 
        plot_bgcolor='#1e293b',  
        font=dict(color='#f8fafc', size=12), 
        title_font=dict(size=16, color='#ffffff'),
        hovermode="closest",
        
        # Alttaki renkli listenin yerleşimi
        legend=dict(
            orientation="h",       
            yanchor="top",
            y=-0.2,               # Yükseklik arttığı için payı biraz daha optimize ettik
            xanchor="center",
            x=0.5,                 
            title_text=f"<b>{hue_col}</b>",
            font=dict(color='#f8fafc', size=11)
        )
    )
    
    fig.update_traces(marker_line_color='#0e1117', marker_line_width=1.5)
    return fig, crosstab_res



def generate_country_treatment_plot(df):
    """
    En çok kayıt içeren ilk 9 ülkeyi ve kalanları 'Other' yaparak
    treatment (tedavi alanların) oranını karanlık tema ile görselleştirir.
    """
    df_country = df[['Country', 'treatment']].dropna().copy()
    
    # 1. En çok veri olan ilk 9 ülkeyi bulalım
    top_9_countries = df_country['Country'].value_counts().nlargest(9).index.tolist()
    
    # 2. İlk 9'da olmayan ülkeleri 'Other' olarak güncelleyelim
    df_country['Country_Grouped'] = df_country['Country'].apply(
        lambda x: x if x in top_9_countries else 'Other'
    )
    
    # 3. Ülke gruplarına göre treatment yüzdelerini hesaplayalım
    # (Önce çapraz tablo yapıp yüzdelik oran çıkartıyoruz)
    ct = pd.crosstab(df_country['Country_Grouped'], df_country['treatment'], normalize='index') * 100
    ct = ct.reset_index()
    
    # Grafikte sadece tedavi görenlerin ('Yes') oranına odaklanalım
    # Eğer kolonda 'Yes' yoksa hata vermemesi için kontrol koyuyoruz
    
    if 1 in ct.columns:
        yes_col = 1
    elif '1' in ct.columns:
        yes_col = '1'
    else:
        # Eğer beklenmedik bir şekilde hala 'Yes' kalmışsa veya fallback durumu için
        yes_col = 1 if 'Yes' in ct.columns else ct.columns[1]
    ct_sorted = ct.sort_values(by=yes_col, ascending=False)

        
    
    # 4. Plotly ile interaktif grafiğimizi çizelim
    fig = px.bar(
        ct_sorted,
        x='Country_Grouped',
        y=yes_col,
        title="<b>Ülkelere Göre Zihinsel Sağlık Tedavisi (Treatment) Oranları (%)</b>",
        labels={yes_col: "Tedavi Görme Oranı (%)", "Country_Grouped": "Ülke"},
        template='plotly_dark',
        color=yes_col,
        color_continuous_scale=px.colors.sequential.Viridis # Orana göre renk değiştiren şık bir palet
    )
    
    # 5. Karanlık tema tasarım ayarları (Dashboard asaletine devam)
    fig.update_layout(
        height=500,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1e293b',
        font=dict(color='#f8fafc', size=12),
        title_font=dict(size=15, color='#ffffff'),
        coloraxis_showscale=False # Yan taraftaki renk skalası barını gizle, kalabalık yapmasın
    )
    
    # Barların üzerine yüzde değerlerini yazalım, fareyle gelmeden de net görünsün
    fig.update_traces(
        texttemplate='%{y:.1f}%', 
        textposition='outside',
        marker_line_color='#0e1117', 
        marker_line_width=1.5
    )
    
    return fig






################################################################
################## ONE_TIME RUN FONKSIYONLARI ##################
################################################################

def merge_files():
    # Load
    new = pd.read_csv('dataset/mental_health.csv')
    old = pd.read_csv('dataset/mental_health_old.csv')

    # Add missing columns to old
    old['UserID'] = range(len(old))
    old['SurveyID'] = '2014'
    change_country(old)  # Update country names in the old dataset
    calculate_turnover_scores(old)  # Calculate turnover risk scores in the old dataset
    kalici_gender_temizligi(old)  # Clean


    # Merge
    merged = pd.concat([new, old], ignore_index=True)

    # Save
    merged.to_csv('dataset/mental_health_combined.csv', index=False)

    print(f"✅ {len(new)} + {len(old)} = {len(merged)} rows merged!")

def change_country(df):

    if 'Country' in df.columns:
        # 2. Country sütunundaki boşlukları temizle
        df['Country'] = df['Country'].astype(str).str.strip()

        # 3. Amerika varyasyonlarını yakalamak için bir harita oluştur
        country_mapping = {
            'United States': 'USA',
            'United States of America': 'USA'
        }

        # 4. Değişiklikleri kalıcı olarak df'e bas (Haritada olmayan diğer ülkeler aynen korunur)
        df['Country'] = df['Country'].map(country_mapping).fillna(df['Country'])
        
        # 5. Orijinal dosyanın üzerine kalıcı olarak yaz
        df.to_csv("dataset/mental_health.csv", index=False) 
        print("✅ İşlem Başarılı! United States varyasyonları 'USA' olarak kalıcı güncellendi.")
    else:
        print("❌ Hata: Veri setinde 'Country' kolonu bulunamadı!")


def calculate_turnover_scores(df):
    # 1. İzin Alma Riski (Leave Risk)
    temp_risk_leave = df['leave'].isin(['Very difficult', 'Somewhat difficult']).astype(int)

    # 2. Şirket Kültürü Riski (Culture Risk)
    temp_risk_culture = df['mental_health_consequence'].isin(['Yes']).astype(int)

    # 3. Tükenmişlik Riski (Burnout Risk)
    temp_risk_burnout = df['work_interfere'].isin(['Often', 'Sometimes']).astype(int)

    # ✨ YENİ: Kaynak ve Bilgi Eksikliği Riski (Resource Risk)
    temp_risk_resources = df['seek_help'].isin(['No', "Don't know"]).astype(int)

    # ✨ YENİ: Damgalanma ve Gizlilik Riski (Anonymity Risk)
    temp_risk_anonymity = df['anonymity'].isin(['No', "Don't know"]).astype(int)

    # ✨ YENİ: Yönetimsel Algı Ayrımcılığı Riski (Perception Risk)
    temp_risk_perception = df['mental_vs_physical'].isin(['No']).astype(int)


    # --- TOPLAM RİSK SKORU HESAPLAMA (0 - 6 Arası) ---
    turnover_risk = (
        temp_risk_leave + 
        temp_risk_culture + 
        temp_risk_burnout + 
        temp_risk_resources + 
        temp_risk_anonymity + 
        temp_risk_perception
    )

    # --- RISK SEGMENTASYONU (Hedef Değişken ve Sınıflandırma) ---
    # Puanı 0-2 ise: Düşük Risk (0)
    # Puanı 3-4 ise: Orta Risk (1)
    # Puanı 5-6 ise: Yüksek İstifa / Tükenmişlik Riski (2)

    conditions = [
        (turnover_risk <= 2),
        (turnover_risk >= 3) & (turnover_risk <= 4),
        (turnover_risk >= 5)
    ]
    #choices = ['Low Risk', 'Medium Risk', 'High Risk']

    #df['turnover_risk'] = np.select(conditions, choices, default='Low Risk')

    # Makine öğrenmesi modeli (XGBoost/Random Forest) için nümerik hedef değişken:
    df['turnover_risk'] = np.select(conditions, [0, 1, 2], default=0)
    df.to_csv("dataset/mental_health.csv", index=False, encoding="utf-8-sig")


def spotify_youtube_mood_classification(df):
    # 1.a) Sayısal sütunlar -> Ortalama ile doldur
#    numeric_cols = ['Valence', 'Energy', 'Danceability', 'Tempo', 'Acousticness', 
#                    'Speechiness', 'Instrumentalness', 'Liveness', 'Loudness']

#    for col in numeric_cols:
#        if col in df.columns:
#            df[col] = df[col].fillna(df[col].mean())

#   # 1.b) Kategorik/Boolean sütunlar -> En sık görülen değer (mod) ile doldur
#    categorical_cols = ['Licensed', 'official_video']
#    for col in categorical_cols:
#        if col in df.columns:
#            # Eğer sütun tamamen boş değilse, mod'unu al, yoksa 0 ata
#            if not df[col].mode().empty:
#                df[col] = df[col].fillna(df[col].mode()[0])
#            else:
#                df[col] = df[col].fillna(0)

    # 1.c) Hala eksik kalan herhangi bir değer varsa (güvenlik önlemi) -> 0 ile doldur
 #   df = df.fillna(0)

#    print(f"✅ Eksik değer kalmadı mı? {df.isnull().sum().sum() == 0}")


    # ---------- 2. PSİKOLOJİK RUHSAL DURUM (MOOD) SINIFLANDIRMASI ----------
    # (13 farklı psikolojik ruh hali)

    conditions = [
        # 1. Bilgilendirici / Eğitici (Podcast, Konuşma)
        (df['Speechiness'] > 0.6) & (df['Instrumentalness'] < 0.1) & (df['Energy'] < 0.6),
        
        # 2. Sakinleştirici / Meditatif
        (df['Energy'] < 0.35) & (df['Acousticness'] > 0.6) & (df['Tempo'] < 100) & (df['Valence'] < 0.5),
        
        # 3. Rahatlatıcı / Huzurlu
        (df['Energy'] < 0.5) & (df['Acousticness'] > 0.5) & (df['Valence'] > 0.6) & (df['Tempo'] < 120),
        
        # 4. Eğitici / İlham Verici
        (df['Speechiness'] > 0.3) & (df['Speechiness'] < 0.6) & (df['Energy'] > 0.4) & (df['Valence'] > 0.5),
        
        # 5. Motivasyonel / Cesaret Verici
        (df['Energy'] > 0.7) & (df['Valence'] > 0.6) & (df['Danceability'] > 0.6) & (df['Tempo'] > 110),
        
        # 6. Melankolik / Düşünsel
        (df['Valence'] < 0.35) & (df['Energy'] < 0.5) & (df['Danceability'] < 0.5) & (df['Acousticness'] > 0.3),
        
        # 7. Enerjik / Canlandırıcı
        (df['Energy'] > 0.8) & (df['Danceability'] > 0.6) & (df['Tempo'] > 120) & (df['Valence'] > 0.4),
        
        # 8. Duygusal / Romantik
        (df['Energy'] < 0.5) & (df['Acousticness'] > 0.4) & (df['Valence'] > 0.3) & (df['Valence'] < 0.6) & (df['Instrumentalness'] < 0.2),
        
        # 9. Düşünsel / Felsefi
        (df['Energy'] > 0.3) & (df['Energy'] < 0.6) & (df['Danceability'] < 0.5) & (df['Tempo'] < 110) & (df['Valence'] < 0.5),
        
        # 10. Huzurlu / Doğal
        (df['Acousticness'] > 0.7) & (df['Energy'] < 0.4) & (df['Instrumentalness'] > 0.3) & (df['Valence'] > 0.4),
        
        # 11. Neşeli / İyimser
        (df['Valence'] > 0.7) & (df['Energy'] > 0.5) & (df['Danceability'] > 0.6) & (df['Speechiness'] < 0.3),
        
        # 12. Hüzünlü / İçe Dönük
        (df['Valence'] < 0.25) & (df['Energy'] < 0.4) & (df['Acousticness'] > 0.5) & (df['Speechiness'] < 0.2),
        
        # 13. Odaklanma / Çalışma (Focus)
        (df['Liveness'] < 0.2) & (df['Energy'] > 0.3) & (df['Energy'] < 0.7) & (df['Instrumentalness'] > 0.2) & (df['Speechiness'] < 0.3)

    ]

    mood_labels = [
        'Informative / Educational',
        'Calming / Meditative',
        'Relaxing / Peaceful',
        'Inspiring / Uplifting',
        'Motivational',
        'Melancholic / Thoughtful',
        'Energetic / Refreshing',
        'Romantic / Emotional',
        'Contemplative',
        'Natural / Serene',
        'Joyful / Optimistic',
        'Sad / Introspective',
        'Focus / Study'
    ]

    # Yeni sütunu oluştur (hiçbir koşula uymayanlara 'Other' ata)
    df['Mood'] = np.select(conditions, mood_labels, default='Other')
    df["Mood"] = df["Mood"].fillna("Other")
    df.loc[df["Mood"] == "", "Mood"] = "Other"

    # ---------- 3. SONUÇLARI İNCELE ----------
    print("\n🔵 Mood (Ruh Hali) Dağılımı:")
    print(df['Mood'].value_counts())

    print("\n📌 İlk 5 satırda Mood sütunu:")
    print(df[['Track', 'Artist', 'Valence', 'Energy', 'Speechiness', 'Mood']].head())

    # Hangi satırlar 'Other' olarak etiketlendi görelim
    print(f"\n⚪ 'Other' olarak etiketlenen satır sayısı: {len(df[df['Mood'] == 'Other'])}")


    # ---------- 4. YENİ VERİ SETİNİ KALICI OLARAK KAYDET ----------
    df.to_csv('spotify_youtube_new.csv', index=False)
    print("\n✅ Yeni veri seti 'spotify_youtube_new.csv' olarak kaydedildi.")


def kalici_gender_temizligi(df):
    print("Veri seti yükleniyor...")
    global gender_check
    if 'Gender' not in df.columns:
        print("❌ Hata: Veri setinde 'Gender' kolonu bulunamadı!")
        return

    # Boşlukları temizle ve geçici olarak küçük harfe çevir
    df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()
    
    # Eşleme Haritası (Aynı anlama gelen varyasyonları yakalar)
    gender_mapping = {
        # === 1. MALE ===
        'm': 'Male', 
        'male': 'Male', 
        'male ': 'Male', 
        'cis male': 'Male', 
        'cis man': 'Male', 
        'masculine': 'Male', 
        'man': 'Male',
        'mal': 'Male', 
        'maile': 'Male', 
        'make': 'Male', 
        'cis_male': 'Male', 
        'mail': 'Male', 
        'malr': 'Male', 
        'masculino': 'Male', 
        'msle': 'Male',
        'cishet male': 'Male',
        'single white male': 'Male',
        'male-ish': 'Male', 
        'guy-ish': 'Male',
        'guy (-ish) ^_^': 'Male',
        'something kinda male?': 'Male',
        'ostensibly male': 'Male',
        'ostensibly male, unsure what that really means': 'Male',
        'male leaning androgynous': 'Male',
        'nb masculine': 'Male',
        'demiguy': 'Male',
        'male (cis)': 'Male',
        
        # === 2. FEMALE ===
        'f': 'Female', 
        'female': 'Female', 
        'female ': 'Female',
        'cis female': 'Female', 
        'cis woman': 'Female', 
        'woman': 'Female', 
        'femme': 'Female', 
        'female-identified': 'Female', 
        'femmina': 'Female', 
        'fm': 'Female', 
        'femail': 'Female', 
        'femake': 'Female',
        'cis-female/femme': 'Female',
        'female assigned at birth': 'Female',
        'female or multi-gender femme': 'Female',
        'female-bodied; no feelings about gender': 'Female',
        'female/gender non-binary.': 'Female',
        'woman-identified': 'Female',
        'female-ish': 'Female',
        'gender non-conforming woman': 'Female',
        'female (cis)': 'Female',
        
        # === 3. NON-BINARY / TRANS ===
        'trans woman': 'Non-Binary / Trans',
        'transfeminine': 'Non-Binary / Trans',
        'trans-female': 'Non-Binary / Trans',
        'female (trans)': 'Non-Binary / Trans',
        'trans female': 'Non-Binary / Trans',
        'transgender woman': 'Non-Binary / Trans',
        'transitioned, m2f': 'Non-Binary / Trans',
        'male-to-female': 'Non-Binary / Trans',
        'mtf': 'Non-Binary / Trans',
        'm2f': 'Non-Binary / Trans',
        'other/transfeminine': 'Non-Binary / Trans',
        'trans man': 'Non-Binary / Trans',
        'male (trans, ftm)': 'Non-Binary / Trans',
        'transgender': 'Non-Binary / Trans',
        'nb': 'Non-Binary / Trans',
        'non-binary': 'Non-Binary / Trans',
        'non binary': 'Non-Binary / Trans',
        'nonbinary': 'Non-Binary / Trans',
        'non-binary and gender fluid': 'Non-Binary / Trans',
        'nonbinary/femme': 'Non-Binary / Trans',
        'genderfluid': 'Non-Binary / Trans',
        'genderfluid (born female)': 'Non-Binary / Trans',
        'genderflux demi-girl': 'Non-Binary / Trans',
        'genderqueer': 'Non-Binary / Trans',
        'genderqueer demigirl': 'Non-Binary / Trans',
        'genderqueer/non-binary': 'Non-Binary / Trans',
        'genderqueer woman': 'Non-Binary / Trans',
        'uhhhhhhhhh fem genderqueer?': 'Non-Binary / Trans',
        'trans non-binary/genderfluid': 'Non-Binary / Trans',
        'enby': 'Non-Binary / Trans',
        'agender trans woman': 'Non-Binary / Trans',
        'agender/genderfluid': 'Non-Binary / Trans',
        'agender': 'Non-Binary / Trans',
        'androgyne': 'Non-Binary / Trans',
        'androgynous': 'Non-Binary / Trans',
        'male/androgynous': 'Non-Binary / Trans',
        'bigender': 'Non-Binary / Trans',
        'fluid': 'Non-Binary / Trans',
        'male (or female, or both)': 'Non-Binary / Trans',
        'male 9:1 female, roughly': 'Non-Binary / Trans',
        'male/genderqueer': 'Non-Binary / Trans',
        'queer/she/they': 'Non-Binary / Trans',
        'she/her/they/them': 'Non-Binary / Trans',
        'pansexual': 'Non-Binary / Trans',
        'p': 'Non-Binary / Trans',
        'queer': 'Non-Binary / Trans',

        # === 4. OTHER / PREFER NOT TO SAY (Absürtler ve Belirtilmeyenler) ===
        '43': 'Other/Prefer Not To Say',
        r'\-': 'Other/Prefer Not To Say',
        '-': 'Other/Prefer Not To Say',
        'other': 'Other/Prefer Not To Say',
        'bilinmiyor': 'Other/Prefer Not To Say',
        'nah': 'Other/Prefer Not To Say',
        'a little about you': 'Other/Prefer Not To Say',
        'other / prefer not to say': 'Other/Prefer Not To Say',
        'god king of the valajar': 'Other/Prefer Not To Say',
        'none of your business': 'Other/Prefer Not To Say',
        'neuter': 'Other/Prefer Not To Say',
        'all': 'Other/Prefer Not To Say',
        'human': 'Other/Prefer Not To Say',
        'i am a wookie': 'Other/Prefer Not To Say',
        'i have a penis': 'Other/Prefer Not To Say',
        'unicorn': 'Other/Prefer Not To Say',
        'sometimes': 'Other/Prefer Not To Say',
        'questioning': 'Other/Prefer Not To Say',
        'nan/empty': 'Other/Prefer Not To Say',
        'afab': 'Other/Prefer Not To Say',
        'contextual': 'Other/Prefer Not To Say',
        'role reversal': 'Other/Prefer Not To Say',
        'rr': 'Other/Prefer not to say',
        'other / prefer not to say': 'Other/Prefer Not To Say'
    }
    df['Gender'] = df['Gender'].replace('\-', 'Other/Prefer not to say')
    df['Gender'] = df['Gender'].map(gender_mapping).fillna(df['Gender'].str.title())

    # 5. Dosyayı orijinalinin üzerine kalıcı olarak kaydet
    df.to_csv("dataset/mental_health.csv", index=False) 
    gender_check = False


def generate_random_timestamp(df):
    try:
        # 1. Sadece Timestamp alanı BOŞ (NaN) VE SurveyID alanı DOLU olan satırların indekslerini alalım
        df['Timestamp'] = df['Timestamp'].astype(object)
        missing_indices = df[df['Timestamp'].isna() & df['SurveyID'].notna()].index
        print(f"Döngüye girecek toplam boş satır sayısı: {len(missing_indices)}")
        
        # 2. Aldığımız indeksler üzerinde for döngüsü başlatıyoruz
        for idx in missing_indices:
            # O satırdaki SurveyID değerini güvenle al ve int'e çevir
            survey_id_val = df.at[idx, 'SurveyID']
            year = int(survey_id_val)
            
            # Rastgele ay ve gün (Takvim hatalarını önleyen limitler)
            month = np.random.randint(1, 13)
            if month in [4, 6, 9, 11]:
                day = np.random.randint(1, 31)
            elif month == 2:
                # Artık yıl kontrolü
                is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
                day = np.random.randint(1, 30 if is_leap else 29)
            else:
                day = np.random.randint(1, 32)
                
            # Rastgele saat, dakika ve saniye
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            
            # String formatında tarihi oluştur
            timestamp_str = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
            
            # .at[indeks, kolon_adı] kullanarak tam o hücreye değeri yazıyoruz
            df.at[idx, 'Timestamp'] = timestamp_str
            
        print("For döngüsü tamamlandı, tüm boş alanlar dolduruldu.")

        df.to_csv("dataset/mental_health.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"Fonksiyon içinde bir hata oluştu: {e}")

def replace_data(df):
    dont_know_cols = [
        'leave',
        'benefits',
        'wellness_program',
        'seek_help',
        'anonymity',
        'mental_vs_physical','care_options','family_history'
    ]

    for col in dont_know_cols:
        if col in df.columns:
            df[col] = df[col].replace({"I don't know": "Don't know"})

    # 'care_options' koloni içindeki 'I am not sure' -> 'Not sure' yapıyoruz
    if 'care_options' in df.columns:
        print("care options replace data yapılıyor....")
        df['care_options'] = df['care_options'].replace(
            {'I am not sure': 'Not sure'}
        )
    
    if 'Gender' in df.columns:
        print("Gender replace data yapılıyor....")
        df['Gender'] = df['Gender'].replace({"Other / Prefer not to say": "Other/Prefer Not To Say"})
    
    df.to_csv('dataset/mental_health.csv', index=False)