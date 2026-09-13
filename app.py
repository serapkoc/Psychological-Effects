import os
import pickle
import sys
import warnings

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import eda as eda
import ab_test as ab
import feature_engineering as fe
import eltv as eltv
import machine_learning as ml
import seaborn as sns
import shap
import llm_services as llm
import n8n as n8n
warnings.filterwarnings(
    "ignore", message=".*FigureCanvasAgg is non-interactive.*"
)

##### UI/UX Ayarları
st.set_page_config(
    page_title="MindTech Insights",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

##### CSS Enjeksiyonu
def local_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    local_css("style.css")
except FileNotFoundError:
    pass

pd.set_option('display.max_columns', None)

TEST_DATA_PATH = "dataset/mental_health_test.csv"

#####  Bağımsız Veri Yükleme (Dizinler Aynen Korundu, Merge Tamamen Çıkarıldı)
@st.cache_data
def load_independent_data():
    try:
        osmi = pd.read_csv("dataset/mental_health.csv")
        spot = pd.read_csv("dataset/spotify_youtube_new.csv")

        if "turnover_risk" not in osmi.columns:
            osmi = eda.calculate_turnover_scores(osmi)
        if "Mood" not in spot.columns:
            spot = eda.spotify_youtube_mood_classification(spot)
        return osmi, spot
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None, None

df_osmi, df_spot = load_independent_data()
st.session_state['df_spot'] = df_spot
#eda.generate_random_timestamp(df_osmi)
if eda.gender_check:
    print(f"Gender Check value {eda.gender_check}")
    eda.kalici_gender_temizligi(df_osmi)


#eda.replace_data(df_osmi)
#df_osmi['Gender'].dtype
#if ((df_osmi['Country'].value_counts() > 0) & (df_osmi['Country'].value_counts() <= 5)).any():
    #fe.kalici_country_temizligi(df_osmi)

#if ((df_osmi['state'].value_counts() > 0) & (df_osmi['state'].value_counts() <= 5)).any():
    #fe.kalici_state_temizligi(df_osmi)

#eda.change_country(df_osmi)
#eda.merge_files()
#print(df_osmi['work_interfere'].unique())
#print(df_osmi['leave'].unique())
#print(df_osmi['no_employees'].unique())
#print(df_osmi['Country'].unique())
#print(df_osmi['Country'].value_counts())
#print((df_osmi['Country'].value_counts() == 1).sum())


#print(df_osmi['Country'].unique())
#print(df_osmi['state'].value_counts())
#print((df_osmi['state'].value_counts() == 1).sum())

#print(df_osmi.describe().T)
#print(df_osmi['SurveyID'].nunique())
#print(df_osmi['SurveyID'].value_counts())
################## MENULER ####################
st.sidebar.title("📌 Navigasyon")

# Stateful Sayfa Yönetimi
if "main_menu" not in st.session_state:
    st.session_state.main_menu = "Genel Bakış"
if "sub_menu" not in st.session_state:
    st.session_state.sub_menu = "Veriye Göz At"

# --- 1. OVERVIEW MENU ---
with st.sidebar.expander("🌐 Genel Bakış", expanded=(st.session_state.main_menu == "Genel Bakış")):
    if st.button("📄 Veriye Göz At", use_container_width=True):
        st.session_state.main_menu = "Genel Bakış"
        st.session_state.sub_menu = "Veriye Göz At"
        st.rerun()
    if st.button("🔬 Veri Işleme(FE)", use_container_width=True):
        st.session_state.main_menu = "Genel Bakış"
        st.session_state.sub_menu = "Veri Işleme(FE)"
        st.rerun()
    if st.button("🚀 Istatistikler", use_container_width=True):
        st.session_state.main_menu = "Genel Bakış"
        st.session_state.sub_menu = "Istatistikler"
        st.rerun()

# --- 2. ÇALIŞAN ANALİZİ MENU ---
with st.sidebar.expander("📊 Çalışan Analizi", expanded=(st.session_state.main_menu == "Çalışan Analizi")):
    if st.button("📄 Kümeleme", use_container_width=True):
        st.session_state.main_menu = "Çalışan Analizi"
        st.session_state.sub_menu = "Kümeleme"
        st.rerun()
    if st.button("🏢 Anomali Tespiti", use_container_width=True):
        st.session_state.main_menu = "Çalışan Analizi"
        st.session_state.sub_menu = "Anomali Tespiti"
        st.rerun()
    if st.button("🤖 Modelleme", use_container_width=True):
        st.session_state.main_menu = "Çalışan Analizi"
        st.session_state.sub_menu = "Modelleme"
        st.rerun()
    if st.button("📈 SHAP & Bias Analizi", use_container_width=True):
        st.session_state.main_menu = "Çalışan Analizi"
        st.session_state.sub_menu = "SHAP & Bias Analizi"
        st.rerun()
    if st.button("🔮 Model Tahminleme", use_container_width=True):
        st.session_state.main_menu = "Çalışan Analizi"
        st.session_state.sub_menu = "Model Tahminleme"
        st.rerun()

# --- 3. İŞ VEREN MENU ---
with st.sidebar.expander("⚙️ İş Veren", expanded=(st.session_state.main_menu == "İş Veren")):
    if st.button("📄 LLM Yorumları", use_container_width=True):
        st.session_state.main_menu = "İş Veren"
        st.session_state.sub_menu = "LLM Yorumları"
        st.rerun()
    if st.button("💡 Öneriler", use_container_width=True):
        st.session_state.main_menu = "İş Veren"
        st.session_state.sub_menu = "Öneriler"
        st.rerun()
    if st.button("🎯 Aksiyon", use_container_width=True):
        st.session_state.main_menu = "İş Veren"
        st.session_state.sub_menu = "Aksiyon"
        st.rerun()

st.sidebar.caption(f"Aktif Sayfa: **{st.session_state.main_menu} > {st.session_state.sub_menu}**")


#ab.change_treatment_type(df_osmi)  # Kalıcı olarak treatment değerlerini 0 ve 1'e dönüştür

osmi_shape = df_osmi.shape if df_osmi is not None else (0, 0)
spot_shape = df_spot.shape if df_spot is not None else (0, 0)



##############################################################
############ EDA - KEŞİFÇİ VERİ ANALİZİ  #####################
##############################################################
if st.session_state.main_menu == "Genel Bakış":

    if st.session_state.sub_menu == "Veriye Göz At":
        REPORT_PICKLE_PATH = "dataset/hatali_ve_aykiri_raporu.pkl"

        st.markdown("### 🌱 MINDTECH INSIGHTS")
        st.markdown("<p class='project-desc'>Teknoloji sektöründe ruh sağlığı, kurumsal kültür ve Spotify&Youtube içerik özellikleri analiz projesi.</p>", unsafe_allow_html=True)

        # Kutucuklu Bilgiler Bölümü
        col_c1, col_c2= st.columns(2)
        with col_c1:
            st.markdown(f"""
            <div class="data-card card-osmi">
                <h5 style="color:#a3bdae; margin:0; font-weight:700;">📊 1. OSMI Ruh Sağlığı</h5>
                <p style="font-size:15px; color:#8fa89b; margin:6px 0;">Çalışanların psikolojik durumları ve sektör algısı.</p>
                <p style="font-size:15px; color:#8fa89b; margin:6px 0;">Amaç: Risk gruplarını, tükenmişlik oranlarını ve tedavi arama eğilimlerini tespit etmek.</p>
                <p style="font-size:16px; color:#6b8275; margin:0 0 6px 0;"><b>Boyut:</b> {osmi_shape[0]:,} Satır / {osmi_shape[1]} Değişken</p>
                <div style="margin-top:4px;"><span class="target-tag" >🎯 Hedef Değişken: treatment, turnover_risk (oluşturuldu) </span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_c2:
            st.markdown(f"""
            <div class="data-card card-spot">
                <h5 style="color:#e0b982; margin:0; font-weight:700;">🎵 3. Spotify & Youtube</h5>
                <p style="font-size:15px; color:#bfa175; margin:6px 0;">Mod düzenleyici ve rahatlatıcı dijital içerikler.</p>
                <p style="font-size:15px; color:#bfa175; margin:6px 0;">Amaç: Çalışanın anlık veya genel psikolojik durumuna uygun içerik eşleştirmesi yapmak.</p>
                <p style="font-size:16px; color:#8c7554; margin:0 0 6px 0;"><b>Boyut:</b> {spot_shape[0]:,} Satır / {spot_shape[1]} Değişken</p>
                <div style="margin-top:4px;"><span class="target-tag" style="background:#33291c; color:#e0b982; border-color:#6e5634;">🎯 Hedef Değişken Yok. Öneri için MOOD değişkeni üretildi</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("📖 📄 1. OSMI Ruh Sağlığı Veri Seti Kolon Sözlüğü & Örnek Veri Yapısı"):
            st.markdown("""
            Aşağıdaki tabloda, `df_osmi` (Kaggle OSMI Mental Health in Tech) veri setindeki tüm sütunların veri tipleri, teknik açıklamaları ve içerdikleri **örnek/benzersiz (unique) değerler** listelenmiştir. Bu tablo, veri temizleme ve modelleme aşamalarında kılavuz olarak kullanılabilir:
            """)
            
            # Şık ve scannable bir dokümantasyon tablosu
            st.markdown("""
                | Sütun Adı | Veri Tipi | Açıklama | Alabileceği Örnek / Benzersiz Değerler |
                | :--- | :--- | :--- | :--- |
                | **`UserID`** | *String* | Katılımcının benzersiz kullanıcı numarası. | `USR_1000`, `USR_1001`, `USR_1002` |
                | **`SurveyID`** | *String* | Anket dönemine ait benzersiz ID. | `SRV_2014` |
                | **`Timestamp`** | *DateTime* | Anketin sisteme kayıt edildiği tarih ve saat. | `2014-08-27 11:29:31`, `2014-08-29 09:12:04` |
                | **`Age`** | *Integer* | Katılımcının beyan ettiği yaş değeri. | `21`, `35`, `44` *(Hatalı/Aykırı değerler içerebilir)* |
                | **`Gender`** | *String* | Katılımcının cinsiyeti. | `Male`, `Female`, `M`, `F`, `Trans-female`, `fluid` |
                | **`Country`** | *String* | Katılımcının yaşadığı ülke. | `United States`, `United Kingdom`, `Canada`, `Germany` |
                | **`state`** | *String* | Katılımcı ABD'deyse yaşadığı eyalet kodu. | `IL`, `CA`, `NY`, `TX`, `NaN` *(ABD dışı için boştur)* |
                | **`self_employed`** | *Kategorik* | Katılımcının kendi işinin sahibi/freelancer durumu. | `Yes`, `No`, `NaN` |
                | **`family_history`** | *Kategorik* | Aile geçmişinde ruhsal rahatsızlık öyküsü. | `Yes`, `No` |
                | **`treatment`** *(🎯)*  | *Kategorik* | **Hedef Değişken:** Ruh sağlığı tedavisi gördü mü? | `Yes`, `No` |
                | **`work_interfere`** | *Kategorik* | Rahatsızlığın iş performansına etkisi. | `Often`, `Sometimes`, `Rarely`, `Never`, `NaN` |
                | **`no_employees`** | *Kategorik* | Çalıştığı şirketteki toplam çalışan sayısı aralığı. | `1-5`, `6-25`, `26-100`, `100-500`, `500-1000`, `1000+` |
                | **`remote_work`** | *Kategorik* | Mesaisinin ne kadarını uzaktan (remote) geçiriyor? | `Yes`, `No` |
                | **`tech_company`** | *Kategorik* | Çalıştığı firma bir teknoloji şirketi mi? | `Yes`, `No` |
                | **`benefits`** | *Kategorik* | Sağlık sigortası ruh sağlığı tedavisini kapsıyor mu?| `Yes`, `No`, `Don't know` |
                | **`care_options`** | *Kategorik* | Şirketin sunduğu bakım seçeneklerini biliyor mu? | `Yes`, `No`, `Not sure` |
                | **`wellness_program`** | *Kategorik* | İşverenin ruh sağlığına yönelik kurumsal programı. | `Yes`, `No`, `Don't know` |
                | **`seek_help`** | *Kategorik* | İşveren destek aramaya yönelik kaynak sunuyor mu? | `Yes`, `No`, `Don't know` |
                | **`anonymity`** | *Kategorik* | Yardım alırsa anonimliğinin korunacağına güveniyor mu?| `Yes`, `No`, `Don't know` |
                | **`leave`** | *Kategorik* | Ruhsal bir sorun için izin almak ne kadar kolay? | `Very easy`, `Somewhat easy`, `Somewhat difficult`, `Very difficult`, `Don't know` |
                | **`mental_health_consequence`** | *Kategorik* | Bunu iş yerinde açmak kariyeri olumsuz etkiler mi? | `Yes`, `No`, `Maybe` |
                | **`phys_health_consequence`** | *Kategorik* | Fiziksel sağlık sorununu açmak olumsuz etkiler mi? | `Yes`, `No`, `Maybe` |
                | **`coworkers`** | *Kategorik* | Sorunlarını iş arkadaşlarıyla paylaşma rahatlığı. | `Yes`, `No`, `Some of them` |
                | **`supervisor`** | *Kategorik* | Sorunlarını doğrudan müdürüyle paylaşma rahatlığı. | `Yes`, `No`, `Some of them` |
                | **`mental_health_interview`** | *Kategorik* | Mülakatta ruh sağlığını açmak önyargı yaratır mı? | `Yes`, `No`, `Maybe` |
                | **`phys_health_interview`** | *Kategorik* | Mülakatta fiziksel sağlığı açmak önyargı yaratır mı?| `Yes`, `No`, `Maybe` |
                | **`mental_vs_physical`** | *Kategorik* | İşveren ruh sağlığını fiziksel kadar ciddi alıyor mu?| `Yes`, `No`, `Don't know` |
                | **`obs_consequence`** | *Kategorik* | İş yerinde sorun yaşayan iş arkadaşlarına şahit oldu mu?| `Yes`, `No` |
                | **`comments`** | *Text* | Katılımcının eklemek istediği serbest görüşler. | *(Metin girdileri veya boş bırakılan alanlar)* |
                | **`turnover_risk`** *(🎯)* | *Kategorik* | **Hedef Değişken:** Şirket iklimine göre hesaplanan istifa/tükenmişlik riski. | `0` (Düşük), `1` (Orta), `2` (Yüksek) |
                """, unsafe_allow_html=True)
        with st.expander("📖 📄 2. Spotify & Youtube Veri Seti Kolon Sözlüğü & Örnek Veri Yapısı"):
            st.markdown("""
            Aşağıdaki tabloda, `df_spot` (Kaggle Spotify & Youtube) veri setindeki tüm sütunların veri tipleri, teknik açıklamaları ve içerdikleri **örnek/benzersiz (unique) değerler** listelenmiştir. Bu tablo, veri temizleme ve modelleme aşamalarında kılavuz olarak kullanılabilir:
            """)
            st.markdown("""
                | Sütun Adı | Veri Tipi | Açıklama | Alabileceği Örnek / Benzersiz Değerler |
                | :--- | :--- | :--- | :--- |
                | **`Unnamed: 0`** (Index) | *Integer* | Satır numarası (indeks). | `0`, `1`, `2`, … |
                | **`Artist`** | *String* | Şarkıcı / grup adı. | `Gorillaz`, `Red Hot Chili Peppers`, `Eminem` | 
                | **`Url_spotify`** | *String* (URL) | Spotify'daki sanatçı sayfası URL'si. | `https://open.spotify.com/artist/3AA28KZvwAUcZuOKwyblJQ` | 
                | **`Track`** | *String* | Şarkı adı. | `Feel Good Inc.`, `Californication`, `Without Me` | 
                | **`Album`** | *String* | Albüm adı. | `Demon Days`, `Californication (Deluxe Edition)` | 
                | **`Album_type`** | *String* | Albüm türü (stüdyo, single, derleme vb.). | `album`, `single`, `compilation` | 
                | **`Uri`** | *String* | Spotify URI (benzersiz tanımlayıcı). | `spotify:track:0d28khcov6AiegSCpG5TuT` | 
                | **`Danceability`** | *Float (0–1)* | Şarkının dans edilebilirlik skoru. | `0.818`, `0.592`, `0.902` | 
                | **`Energy`** | *Float (0–1)* | Şarkının enerji seviyesi. | `0.705`, `0.767`, `0.72` | 
                | **`Key`** | *Integer (0–11)* | Müzikal ton (C, C#, … B). | `6`, `9`, `0` | 
                | **`Loudness`** | *Float (dB)* | Ses yüksekliği (desibel). | `-6.679`, `-2.788`, `-9.62` | 
                | **`Speechiness`** | *Float (0–1)* | Konuşma oranı (vokalin sözlü olma derecesi). | `0.177`, `0.027`, `0.0541` | 
                | **`Acousticness`** | *Float (0–1)* | Akustik olma derecesi. | `0.00836`, `0.0021`, `0.0173` | 
                | **`Instrumentalness`** | *Float (0–1)* | Enstrümantal olma oranı. | `0.00233`, `0.00165`, `0.0436` | 
                | **`Liveness`** | *Float (0–1)* | Canlı kayıt olma olasılığı. | `0.613`, `0.127`, `0.0414` | 
                | **`Valence`** | *Float (0–1)* | Müziğin pozitiflik/neşe derecesi. | `0.772`, `0.328`, `0.884` | 
                | **`Tempo`** | *Float (BPM)* | Şarkının dakikadaki vuruş sayısı. | `138.559`, `96.483`, `117.002` | 
                | **`Duration_ms`** | *Integer (ms)* | Şarkı süresi (milisaniye). | `222640`, `329733`, `294227` | 
                | **`Url_youtube`** | *String (URL)* | YouTube video URL'si. | `https://www.youtube.com/watch?v=HyHNuVaZJ-k` | 
                | **`Title`** | *String* | YouTube video başlığı. | `Gorillaz - Feel Good Inc. (Official Video)` | 
                | **`Channel`** | *String* | YouTube kanal adı. | `Gorillaz`, `Red Hot Chili Peppers` | Hayır |
                | **`Views`** | *Integer* | YouTube video izlenme sayısı. | `693555221`, `1018811259`, `1682616458` | 
                | **`Likes`** | *Integer* | YouTube beğeni sayısı. | `6220896`, `4394471`, `10481678` | 
                | **`Comments`** | *Integer* | YouTube yorum sayısı. | `169907`, `121452`, `296745` | 
                | **`Description`** | *String (Text)* | YouTube video açıklaması. | Uzun metin (şarkı sözleri, künye vb.) | 
                | **`Licensed`** | *Boolean* | Video resmi lisanslı mı? | `True`, `False` | 
                | **`official_video`** | *Boolean* | Video resmi müzik videosu mu? | `True`, `False` | 
                | **`Stream`** | *Integer* | YouTube'daki akış (stream) sayısı. | `1040234854`, `1055738398`, `1041736808` | 
                | **`Mood`** | *String* | Spotify verilerinden üretilen mod etiketi | `Informative / Educational`,`Calming / Meditative`,`Relaxing / Peaceful` | 
                """, unsafe_allow_html=True)
        # 6. Veri Setleri Alt Alta Listeleme (Merge Yok, Ham CSV Listesi)
        if df_osmi is not None:
            df_pickle = pd.DataFrame(eda.load_pickle_file())  # OSMI veri setindeki hatalı ve aykırı değerleri analiz et ve Pickle olarak kaydet
            

            if df_pickle is None or df_pickle.empty:
                df_pickle = pd.DataFrame(eda.analyze_incorrect_osmi_datas(df_osmi))
                print("OSMI veri seti hataları analiz edildi ve Pickle dosyası oluşturuldu.")
      

            if df_pickle is not None or not df_pickle.empty:
                st.markdown("---")
                st.markdown("### 📊 Veri Seti Boş,Negatif,Aykırı Değer Özet Raporu ")
                #st.markdown(f"""Pickle dosyası yolu: `{df_pickle}`""")

                # Yan yana 3 adet kutu (kolon)
                col_null, col_neg, col_age = st.columns(3)

                # --- 1. KUTU: BOŞ DEĞERLER ---
                with col_null:
                    st.markdown("#### ⚪ Boş (NaN) Değerler")
                    null_df = df_pickle[df_pickle["Hata Türü"] == "NaN"]
                    st.metric(label="Toplam Boş Hücre", value=len(null_df))
                    
                    counts = null_df["Kolon Adı"].value_counts()
                    items = [f" **{k_adi}**: {adet} adet" for k_adi, adet in counts.items()]

                    # Aralarına ' | ' koyarak tek bir satırda yazdırıyoruz
                    st.write(" | ".join(items))
                # --- 2. KUTU: NEGATİF DEĞERLER ---
                with col_neg:
                    st.markdown("#### 🔴 Negatif Değerler")
                    neg_df = df_pickle[df_pickle["Hata Türü"] == "Negative"]
                    st.metric(label="Toplam Negatif Girdi", value=len(neg_df))
                    unique_negatif = neg_df["Mevcut Değer"].astype(str).unique()
                    st.write(f"🚨 Saptanan Negatifler: `{', '.join(unique_negatif)}`")

                    counts = neg_df["Kolon Adı"].value_counts()

                    for k_adi, adet in counts.items():
                        st.write(f"📌 **{k_adi}**: {adet} adet")

                # --- 3. KUTU: AYKIRI YAŞLAR ---
                with col_age:
                    st.markdown("#### 🧓 Aykırı Yaş Değerleri")
                    age_df = df_pickle[df_pickle["Hata Türü"] == "Outlier"]
                    st.metric(label="Toplam Aykırı Yaş", value=len(age_df))
                    
                    #unique_ages = age_df["Mevcut Değer"].unique()
                    unique_ages = age_df["Mevcut Değer"].astype(str).unique()
                    st.write(f"🚨 Saptanan Girdiler: `{', '.join(unique_ages)}`")
            else:
                st.markdown("<div class='section-divider'>Negatif, Boş ve Aykırı Değerleri okurken hata oluştu!!</div>", unsafe_allow_html=True)
            
            ######################## Verilerin Grafiksel Gösterimi ########################
            available_columns = eda.available_columns(df_osmi)
                
            ############ Tüm değişkenler Arası Cross  Control#################
            ##################################################################
            col1, col2 = st.columns(2)
            with col1:
                x_axis_col = st.selectbox(
                    "📌 Ana Değişken (Grafiğin Alt Ekseni):", 
                    options=available_columns,
                    index=available_columns.index('remote_work') if 'remote_work' in available_columns else 0
                )
            with col2:
                hue_col = st.selectbox(
                    "🎨 Renklendirme / Alt Kırılım Değişkeni:", 
                    options=available_columns,
                    index=available_columns.index('Gender') if 'Gender' in available_columns else 1
                )
                
            # Koruma bariyeri
            if x_axis_col == hue_col:
                st.warning("⚠️ Analizin anlamlı olması için lütfen iki farklı değişken seçin.")
            
            # Grafik ve Tablo üretimi
            #fig, crosstab_res = eda.generate_cross_plot(df_osmi, x_axis_col, hue_col)
            fig, crosstab_res = eda.generate_cross_plotly(
                df_osmi, x_axis_col, hue_col
            )
            
            # Grafiği ekrana basma
            #st.pyplot(fig)
            st.plotly_chart(fig, use_container_width=True)

            #st.write(f"📊 **{hue_col}** Filtreleme Menüsü (Açıp/Kapatmak için tıklayın):")

            ############# Kİ-KARE TESTİ ARAYÜZÜ #########################
            st.subheader("🎯 Ki-Kare Bağımsızlık Testi Sonuçları")
            st.write("Faktörlerin **Treatment** (Tedavi) ve **Turnover Risk** (İşten Ayrılma Riski) ile olan ilişkilerinin karşılaştırmalı analizi:")
            # Fonksiyonu çağırıp tabloyu basıyoruz
            double_chi2_df = ab.run_chi2_tests_double(df_osmi)
            st.dataframe(double_chi2_df, width="stretch", hide_index=True)


            # --- ÜLKE BAZLI GRAFİK ARAYÜZÜ ---
            st.subheader("🌍 Ülke Bazlı Tedavi (Treatment) Oranları Analizi")
            st.write("En çok veri girişi yapılan ilk 9 ülke ve diğer ('Other') ülkelerin tedavi görme yüzdeleri:")

            # Fonksiyonu çağırıp Plotly grafiğini basıyoruz
            country_fig = eda.generate_country_treatment_plot(df_osmi)
            st.plotly_chart(country_fig, use_container_width=True)


            ############################ A/B TESTİ ARAYÜZÜ ###############################
            st.markdown("## 🧪 Gelişmiş A/B Testing Laboratuvarı")
            st.write("""
            **Ana Hipotez:** *Remote (Uzaktan) çalışmanın çalışan motivasyonuna olumlu etkisi vardır.*  
            Aşağıdaki açılır menüden tekil risk faktörlerini seçebilir veya **Burnout Endeksi** ile her iki riskin birleştiği büyük resmi inceleyebilirsiniz:
            """)

            # 🎯 TAM İSTEDİĞİN O EFSANE DROPDOWN (SELECTBOX) YAPISI:
            metrik_secimi = st.selectbox(
                "📊 Yarıştırılacak Metrik veya Endeksi Seçin:",
                options=['treatment', 'turnover_risk', 'combined'],
                format_func=lambda x: {
                    'treatment': "🚨 Zihinsel Sağlık Tedavisi Oranı (treatment = 1)",
                    'turnover_risk': "📉 İşten Ayrılma Eğilimi Oranı (turnover_risk = 2)",
                    'combined': "🔥 Tükenmişlik / Burnout Endeksi (treatment=0 & turnover_risk=2)"
                }[x]
            )

            # Seçilen moda göre esnek A/B fonksiyonumuzu çağırıyoruz
            ab_fig, ab_report = ab.run_flexible_ab_test(df_osmi, mode=metrik_secimi)

            # Ekrana basma
            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.markdown(ab_report)

            with col2:
                st.plotly_chart(ab_fig, use_container_width=True)

            ################  Yıllara göre tedavi anlamlılığı ###########################
            test_results = ab.run_chi2_year_treatment(df_osmi)
            
            col1, col2 = st.columns([1, 1], gap="large")
            with col1:
                # Basit Dağılım Tablosu
                st.subheader("📋 Yıllara Göre Tedavi Alma/Almama Durumları")
                ct_table = test_results["table"].copy()
                ct_table.columns = ['Tedavi Almayan (0)', 'Tedavi Alan (1)']
                ct_table['Toplam Satır'] = ct_table['Tedavi Almayan (0)'] + ct_table['Tedavi Alan (1)']
                st.dataframe(ct_table)

            with col2:
                # Net İstatistiksel Sonuç
                st.subheader("🎯 İstatistiksel Sonuç")
                st.write(f"**p-değeri:** {test_results['p_value']:.4e}")
                if test_results["significant"]:
                    st.error("🔴 **Sonuç:** Yıllara göre treatment oranlarında istatistiksel olarak **anlamlı bir fark VARDIR**.")
                else:
                    st.success("🟢 **Sonuç:** Yıllara göre treatment oranlarında istatistiksel olarak **anlamlı bir fark YOKTUR**.")
            
            ####################### Verilerin Özet Görselleştirmesi #######################
            st.markdown("#### 📄 OSMI Veri Seti (`mental_health.csv`)")
            st.dataframe(df_osmi.head(10), width="stretch")
            st.markdown("#### 🎵 Spotify & YouTube Veri Seti (`spotify_youtube.csv`)")
            st.dataframe(df_spot.head(10), width="stretch")

    elif st.session_state.sub_menu == "Veri Işleme(FE)":

        FE_PICKLE_PATH = "dataset/fe_summary_file.pkl"

        st.set_page_config(page_title="Feature Engineering Özet Paneli", layout="wide")

        st.title("📊 Feature Engineering (FE) Özet Sayfası")
        st.caption("Veri ön işleme ve özellik mühendisliği adımlarının canlı metadatası")
        st.divider()

        ############ ÖZET VERİ İÇİN PİCKLE OLUŞTURMA ##################
        summary = fe.fe_summary_pickle(df_osmi)

        if summary is not None:

            # --- TABLOLAR VE GÖRÜNÜM ---
            
            # [1] Eksik Veri Doldurma Paneli
            st.header("1️⃣ Eksik Veri Doldurma (Imputation) Stratejileri")
            if summary["missing_values"]:
                df_missing = pd.DataFrame(summary["missing_values"])
                st.dataframe(df_missing, width="stretch", hide_index=True)
            else:
                st.success("Ham veride eksik veri içeren kategorik sütun bulunamadı.")
                
            st.divider()

            # [2] Aykırı Değer Paneli
            st.header("2️⃣ Aykırı Değer Yönetimi (Outlier Handling)")
            o = summary["outliers"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Hedef Sütun", value=o["Sütun"])
            with col2:
                st.metric(label="Yakalanan Aykırı Satır Sayısı", value=f"{o['Aykırı_Adet']} adet")
            with col3:
                st.metric(label="Atanan Yeni Medyan Değeri", value=o["Atanan_Medyan"])
                
            st.info(f"**Uygulanan Kriter ve Yöntem:** {o['Kriter']} yaş aralığı dışındakiler yakalanarak **{o['Yöntem']}** tekniği ile düzeltilmiştir.")
            st.divider()

            # [3] Yeni Özellikler Paneli
            st.header("3️⃣ Yeni Türetilen Özellikler (Feature Extraction)")
            df_extract = pd.DataFrame(summary["extracted_features"])
            st.dataframe(df_extract, width="stretch", hide_index=True)
            st.divider()

            # [4] Silinen Özellikler Paneli
            st.header("4️⃣ Veri Setinden Çıkarılan Sütunlar (Feature Dropping)")
            df_drop = pd.DataFrame(summary["dropped_features"])
            st.dataframe(df_drop, width="stretch", hide_index=True)

            st.success("💡 Bu sayfadaki veriler dondurulmuş bir şekilde yerel `.pkl` dosyasından okunduğu için transformasyonlar tamamlansa bile geçmiş kayıpları önler.")
        else:
            st.info("HATA OLUŞTU: Feature Engineering özet dosyası bulunamadı veya okunamadı. Lütfen veri setini kontrol edin ve FE adımlarını yeniden çalıştırın.")

        ############ TRAIN/ TEST AYRIŞTIRMA ##################
        TRAIN_CSV_PATH = "dataset/mental_health_train.csv"
        TEST_CSV_PATH = "dataset/mental_health_test.csv"

        # 2. KONTROL: Dosya diskte zaten var mı?
        if (os.path.exists(TRAIN_CSV_PATH) and os.path.exists(TEST_CSV_PATH)):
            # --- VARSA ÇALIŞACAK BLOK ---
            print(f"🔄 Temizlenmiş master dosya bulundu! Doğrudan diskten okunuyor: {TRAIN_CSV_PATH}")
            df_mental_health_train = pd.read_csv(TRAIN_CSV_PATH)
            df_mental_health_test = pd.read_csv(TEST_CSV_PATH)

        else:
            df_xtrain, df_xtest, df_ytrain, df_ytest = fe.split_train_test(df_osmi)
            print(f"Train seti boyutu: {df_xtrain.shape[0]}, Test seti boyutu: {df_xtest.shape[0]}")
        
            st.session_state.y_train = df_ytrain
            st.session_state.y_test = df_ytest

            ############ BOŞ DEĞERLERİ DOLDURMA ##################
            df_xtrain_missing, df_xtest_missing = fe.fill_missing_values(df_xtrain, df_xtest)  # Eksik değerleri doldurmak için Feature Engineering fonksiyonunu çağır
            print(f"Boş değerler dolduruldu ve aykırı değerler işlendi. Train seti boyutu: {df_xtrain_missing.shape[0]}, Test seti boyutu: {df_xtest_missing.shape[0]}")

            ############ FEATURE EXTRACTION ##################
            df_xtrain_extracted, df_xtest_extracted = fe.features_extract(df_xtrain_missing, df_xtest_missing)
            print(f"Yeni özellikler eklendi. Train seti boyutu: {df_xtrain_extracted.shape[0]}, Test seti boyutu: {df_xtest_extracted.shape[0]}")

            ############ FEATURE DROP ##################
            df_xtrain_dropped, df_xtest_dropped = fe.features_drop(df_xtrain_extracted, df_xtest_extracted)
            print(f"Bazı özellikler çıkarıldı. Train seti boyutu: {df_xtrain_dropped.shape[0]}, Test seti boyutu: {df_xtest_dropped.shape[0]}")

            df_xtrain_final, df_xtest_final = fe.apply_encoding_and_scaling(df_xtrain_dropped, df_xtest_dropped)
            print(f"Encoding & Scale işlemleri tamamlandı. Train seti boyutu: {df_xtrain_final.shape[0]}, Test seti boyutu: {df_xtest_final.shape[0]}")

            st.session_state.X_train = df_xtrain_final
            st.session_state.X_test = df_xtest_final

    elif st.session_state.sub_menu == "Istatistikler":

        #1. Veriyi okuyup eltv.py ile hesaplıyorsun
        df_osmi_eltv = eltv.calculate_eltv(df_osmi)

        summary = eltv.get_eltv_kpi_summary(df_osmi_eltv)

        # 3. Tepedeki 4'lü Kutu / Metric Düzeni
        st.subheader("📊 ELTV Segment Özet Göstergeleri")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="🟢 Yüksek Bağlılık",
                value=f"{summary['Yüksek Bağlılık (High)']['count']} Kişi",
                delta=f"%{summary['Yüksek Bağlılık (High)']['pct']:.1f}",
                delta_color="normal",
            )

        with col2:
            st.metric(
                label="🟡 Orta Bağlılık",
                value=f"{summary['Orta Bağlılık (Medium)']['count']} Kişi",
                delta=f"%{summary['Orta Bağlılık (Medium)']['pct']:.1f}",
                delta_color="off",
            )

        with col3:
            st.metric(
                label="🟠 Düşük Bağlılık",
                value=f"{summary['Düşük Bağlılık (Low)']['count']} Kişi",
                delta=f"%{summary['Düşük Bağlılık (Low)']['pct']:.1f}",
                delta_color="inverse",
            )

        with col4:
            st.metric(
                label="🔴 Kritik / Riskli",
                value=f"{summary['Kritik / Riskli (Critical)']['count']} Kişi",
                delta=f"%{summary['Kritik / Riskli (Critical)']['pct']:.1f}",
                delta_color="inverse",
            )

        st.divider()  # Altına çizgi çekip matris grafiğini başlatırsın

        # 2. Grafiği basıyorsun
        col_left, col_center, col_right = st.columns([1, 4, 1])

        with col_center:
            fig_matrix = eltv.plot_eltv_4_segment_matrix(df_osmi_eltv)
            st.pyplot(fig_matrix, use_container_width=False)
        
        st.header("🚨 Risk & Stigma (Damgalanma) Analizi")

        # 1. Hesaplamayı Yap ve Sıralı Veriyi Al
        df_risk_sorted = eltv.calculate_risk_and_stigma_indices(df_osmi)

        # 2. Tepede Kritik Durum KPI'ları
        col1, col2, col3 = st.columns(3)

        high_burnout_count = len(df_risk_sorted[df_risk_sorted['Burnout_Risk_Score'] >= 70])
        high_stigma_count = len(df_risk_sorted[df_risk_sorted['Stigma_Index'] >= 70])

        with col1:
            st.metric(label="🔥 Yüksek Burnout Riski (Score ≥ 70)", value=f"{high_burnout_count} Çalışan")

        with col2:
            st.metric(label="🛡️ Yüksek Stigma Korkusu (Index ≥ 70)", value=f"{high_stigma_count} Çalışan")

        with col3:
            st.metric(
                label="📊 Ort. Burnout Skoru",
                value=f"{df_risk_sorted['Burnout_Risk_Score'].mean():.1f} / 100"
            )

        st.subheader("📋 En Yüksek Risk Altındaki Çalışanlar (Azalan Sırayla)")

        # Tabloda gösterilecek kritik kolonları seçelim
        display_cols = [
            'Burnout_Risk_Score',
            'Stigma_Index',
            'work_interfere',
            'leave',
            'turnover_risk'
        ]

        # Mevcut kolonlardan var olanları güvenle filtreele
        available_display_cols = [col for col in display_cols if col in df_risk_sorted.columns]

        # Streamlit Dataframe Gösterimi ve Renklendirme
        st.dataframe(
            df_risk_sorted[available_display_cols].head(75), # En riskli ilk 50 kişi
            width="stretch",
            column_config={
                "Burnout_Risk_Score": st.column_config.ProgressColumn(
                    "Burnout Risk Skoru",
                    help="0-100 arası tükenmişlik riski",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "Stigma_Index": st.column_config.ProgressColumn(
                    "Stigma Korkusu Endeksi",
                    help="0-100 arası damgalanma çekincesi",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
            }
        )

elif st.session_state.main_menu == "Çalışan Analizi":
    print("Çalışan Analizi Menüsü")
    if st.session_state.sub_menu == "Kümeleme":
        print("Makine Öğrenmesi işlemleri - Gruplama Menüsü")

        if 'df_mldrop' not in st.session_state:
            df_mldrop = ml.ml_features_drop(df_osmi)
            st.session_state['df_mldrop'] = df_mldrop
        else:
            df_mldrop = st.session_state['df_mldrop']

        # 2. Missing Values Fill
        if 'df_mlfill' not in st.session_state:
            df_mlfill = ml.ml_fill_missing_values(df_mldrop)
            st.session_state['df_mlfill'] = df_mlfill
        else:
            df_mlfill = st.session_state['df_mlfill']

        # 3. Nominal to Ordinal Mapping
        if 'df_mlmapping' not in st.session_state:
            df_mlmapping = ml.nominal_to_ordinal(df_mlfill)
            st.session_state['df_mlmapping'] = df_mlmapping
            st.session_state['scaled_df'] = df_mlmapping  # Özel session adın
        else:
            df_mlmapping = st.session_state['df_mlmapping']

        # 4. Unsupervised Scaling
        if 'df_mlscale' not in st.session_state:
            df_mlscale = ml.unsupervised_scaling(df_mlmapping)
            st.session_state['df_mlscale'] = df_mlscale
        else:
            df_mlscale = st.session_state['df_mlscale']

        # 5. K-Means Calculation (Çoklu dönüş değerleri için)
        if 'kmeans_results' not in st.session_state:
            fig_elbow, silhouette_vis, optimum_k, df_pca, total_var = ml.calculate_kmeans(df_mlscale)
            st.session_state['kmeans_results'] = {
                'fig_elbow': fig_elbow,
                'silhouette_vis': silhouette_vis,
                'optimum_k': optimum_k,
                'df_pca': df_pca,
                'total_var': total_var
            }
        else:
            results = st.session_state['kmeans_results']
            fig_elbow = results['fig_elbow']
            silhouette_vis = results['silhouette_vis']
            optimum_k = results['optimum_k']
            df_pca = results['df_pca']
            total_var = results['total_var']
        print(f"PCA - Total Var: %{total_var})")

        #st.pyplot(fig_elbow)
        #st.pyplot(silhouette_vis)

        st.write(f"### Optimal Küme Sayısı (K): {optimum_k}")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.pyplot(fig_elbow)
        with col2:
            st.pyplot(silhouette_vis)

        with col3:
            kmeans = KMeans(n_clusters=optimum_k, random_state=42, n_init='auto')
            df_pca['Cluster'] = kmeans.fit_predict(df_pca)
            fig_pca, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_pca,
                x='PC1',
                y='PC2',
                hue='Cluster',
                palette='Set1',
                ax=ax,
                alpha=0.8,
            )

            ax.set_title(
                f"PCA ile K-Means Kümeleme (Korunan Bilgi: %{total_var:.1f})", fontsize=12
            )
            #ax.set_xlabel(f"PC1 ({total_var[0]:.1f}% varyans)")
        # ax.set_ylabel(f"PC2 ({total_var[1]:.1f}% varyans)")
            st.pyplot(fig_pca)
        plt.close()
        

        # 1. Kümeleri orijinal verimizle birleştirip özet çıkarıyoruz
        df_with_clusters, cluster_summary = ml.get_cluster_profiles(df_mlfill, df_pca, kmeans)

        st.write('---')
        st.subheader('📊 Küme Profilleri ve İstatistikleri')

        # Her kümedeki kişi sayısını ve ortalamaları gösterelim
        st.dataframe(cluster_summary)

        # 2. Kümelere Anlamlı İsimler Verelim (Örnek Mantık)
        # Arayüzdeki tablolara bakarak bu isimleri veri setindeki baskın özelliklere göre özelleştirebilirsin:
        cluster_names = {
            0: '🟢 Küme 0: Düşük Riskli / Destek Alanlar',
            1: '🟡 Küme 1: Orta Riskli / Farkındalığı Yüksek Olanlar',
            2: '🔴 Küme 2: Yüksek Riskli / Destek Almayan Yoğun Stres Grubu',
            3: '🔵 Küme 3: Semptom Gösteren / Kurumsal Destek Bekleyenler',
        }

        # Verisetindeki sayısal '0, 1, 2, 3' etiketlerini bu isimlerle değiştirelim
        df_with_clusters['Cluster_Name'] = df_with_clusters['Cluster'].map(
            cluster_names
        )

        st.success('Kullanıcılar başarıyla 4 gruba atandı!')

        # 3. Örnek Kullanıcı Listesi Gösterimi
        st.write('### 👤 Örnek Kullanıcı Atamaları')
        st.dataframe(df_with_clusters[['Age', 'Gender', 'Cluster_Name']].head(10))  # İlgili sütunları seçebilirsin
        st.session_state.df_with_clusters = df_with_clusters
        st.session_state.kmeans_ready = True
        ############### HIYERARŞİK KUMELEME ######################

        st.write("---")
        st.header("🌳 Hiyerarşik Kümeleme (Hierarchical Clustering)")
        # K-Means'teki optimal K değerini varsayılan yapabilir veya kullanıcıya seçtirebiliriz
        selected_k = st.slider(
            "Dendrograma göre bir küme sayısı (K) seçin:",
            min_value=2,
            max_value=8,
            value=4,  # K-Means ile aynı olsun diye varsayılan 4
        )
        # 1. Dendrogram Grafiğini Ekrana Basma
        col1, col2= st.columns(2)

        with col1:
            st.subheader("1. Dendrogram (Ağaç Grafiği)")
            st.write(
                "Aşağıdaki ağaç yapısı verilerin aşağıdan yukarıya nasıl birleştiğini gösterir. "
                "En uzun dikey çizgileri keserek ideal küme sayısına karar verebilirsin."
            )

            # Dendrogram fonksiyonunu çağırıyoruz (ölçeklenmiş/PCA verini vererek)
            fig_dendrogram = ml.plot_dendrogram(df_pca,n_clusters=selected_k)
            st.pyplot(fig_dendrogram)
            plt.close(fig_dendrogram)

        with col2:
            # 2. Küme Sayısını Seçme ve Modeli Çalıştırma
            st.subheader("2. Hiyerarşik Kümeleme Sonuçları")
            # Model çalıştırma
            st.markdown("<br><br>", unsafe_allow_html=True)        
            hierarchical_labels, hc_model = ml.calculate_hierarchical(df_pca, n_clusters=selected_k)

            # Etiketleri dataframe'e ekleme
            df_hc_result = df_pca.copy()
            df_hc_result["HC_Cluster"] = hierarchical_labels

            # 3. Hiyerarşik Kümeleme Scatter Plot (2D Görselleştirme)
            fig_hc, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=df_hc_result,
                x="PC1",
                y="PC2",
                hue="HC_Cluster",
                palette="Set2",
                ax=ax,
                alpha=0.8,
            )
            ax.set_title(
                f"Hiyerarşik Kümeleme Sonucu (K={selected_k})", fontsize=12
            )
            st.pyplot(fig_hc)
            plt.close(fig_hc)

        st.write("SONUÇ: Hiyerarşik Kümeleme algoritması serbest (distance threshold) bırakıldığında, en yüksek mesafe "
                "farkıyla veriyi doğal olarak 3 ana gruba ayırmaktadır. Ancak K-Means modelimizdeki 4'lü yapıyla kıyaslama " \
                "yapabilmek ve veriyi bir tık daha detaylı segmente edebilmek adına kesim seviyesi Mesafe $\approx 73$ " \
                "seviyesine çekilerek 4'lü kümeleme yapısı da başarıyla elde edilmiştir.")

    elif st.session_state.sub_menu == "Anomali Tespiti":
        print("Anamoli testi ve sessiz istifa analizi menüsü")
        st.header('🚨 Modül 7: Anomali Testi' \
        'piti & Sessiz İstifa (Quiet Quitting) Analizi')
        st.markdown(
            """
            Bu modülde gözetimsiz öğrenme (**Isolation Forest** & **LOF**) yöntemleri kullanarak:
            1. **Anket Manipülasyonu:** Anket sorularını rastgele/sallayarak dolduran yanıtları,
            2. **Sessiz İstifa:** Dışarıya mutlu gözüküp davranışsal olarak şirketle bağı kopmuş çalışanları tespit ediyoruz.
        """
        )

        # Arayüz Parametreleri
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            contamination_rate = st.slider(
                'Tahmini Anomali/Sessiz İstifa Oranı (%)',
                min_value=1,
                max_value=15,
                value=5,
                step=1,
            )
        with col_p2:
            selected_method = st.selectbox(
                'Algoritma Seçimi',
                options=['both', 'isolation_forest', 'lof'],
                format_func=lambda x: {
                    'both': 'Her İkisi (Konsensus - En Güvenilir)',
                    'isolation_forest': 'Isolation Forest',
                    'lof': 'Local Outlier Factor (LOF)',
                }[x],
            )
        print(f"Seçilen method: {selected_method} ")
        if st.button('🔍 Anomalileri ve Şüpheli Yanıtları Tespit Et'):
            if ("scaled_df" in st.session_state and st.session_state.scaled_df is not None):
                X_train = st.session_state.scaled_df
            else:
                df_mldrop = ml.ml_features_drop(df_osmi)
                df_mlfill = ml.ml_fill_missing_values(df_mldrop)
                X_train = ml.nominal_to_ordinal(df_mlfill)

            with st.spinner('Gözetimsiz öğrenme algoritmaları çalıştırılıyor...'):
                # 1. Anomali Fonksiyonunu Çağırıyoruz
                df_anomalies, score_col = ml.detect_anomalies(
                    X_train,
                    contamination=contamination_rate / 100.0,
                    method=selected_method,
                )

                anomalies_only = df_anomalies[
                    df_anomalies['is_anomaly'] == 'Şüpheli / Anomali'
                ]
                normal_only = df_anomalies[df_anomalies['is_anomaly'] == 'Normal']

            # Özet Metrikler
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric('Toplam İncelenen Yanıt', len(df_anomalies))
            m2.metric(
                'Tespit Edilen Anomali / Şüpheli',
                len(anomalies_only),
                delta=f'%{len(anomalies_only)/len(df_anomalies)*100:.1f}',
                delta_color='inverse',
            )
            m3.metric('Normal Yanıt', len(normal_only))

            # --- GÖRSELLEŞTİRME: Anomali Dağılımı ---
            st.write('### 📊 1. Anomali Skor Dağılımı')
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(
                data=df_anomalies,
                x=score_col,
                hue='is_anomaly',
                element='step',
                palette={'Normal': '#2ecc71', 'Şüpheli / Anomali': '#e74c3c'},
                ax=ax,
            )
            # x='iso_score',

            plt.xlabel('Anomali Skoru (Düşük Skor = Yüksek Anomali Riski)')
            plt.ylabel('Çalışan Sayısı')
            st.pyplot(fig)

            # --- DETAYLI İNCELEME VE İŞ MANTIĞI (QUIET QUITTING vs MANİPÜLASYON) ---
            st.write('### 🕵️‍♂️ 2. Şüpheli Profil İncelemeleri (İlk 10 Kayıt)')
            st.markdown(
                'Aşağıdaki tabloda anomali olarak etiketlenmiş çalışanların öne çıkan özellikleri yer almaktadır:'
            )

            # Anomali olan çalışanların listesi
            st.dataframe(anomalies_only.head(10), width='stretch')

            # --- SESSİZ İSTİFA / MANİPÜLASYON İPUÇLARI ---
            st.info(
                """
            💡 **IK ve Yönetim İçin Aksiyon Önerileri:**
            * **Anket Manipülasyonu Tespiti:** Bir çalışanın `stigma_score` veya `leave` gibi kritik alanlarda uç değerlerde gezindiği ancak diğer skorlarının aşırı nötr olduğu durumlar anketin hızlıca geçiştirildiğini gösterir. Bu veriler modellerin eğitim setinden çıkarılabilir.
            * **Sessiz İstifa (Quiet Quitting) Tespiti:** Çalışanın iş memnuniyeti/destek skorları yüksek görünmesine rağmen, sosyal/kurumsal aktivitelere katılım ve destek arayışı parametrelerinde tamamen soyutlandığı görülen profiller doğrudan IK görüşmesine yönlendirilmelidir.
            """
            )

    elif st.session_state.sub_menu == "Modelleme":

        X_train, y_train_treatment, y_train_turnover = ml.read_train()

        if X_train is not None:
            st.session_state["y_train_treatment"] = y_train_treatment
            st.session_state["y_train_turnover"] = y_train_turnover

            #ml.ml_cross_validation(X_train, y_train_treatment,y_train_turnover)
            print('--- 1. Treatment Modelleri Yarışıyor ---')
            st.subheader("1️⃣ Baseline Modellerin Cross-Validation Yarışı (Varsayılan Parametreler)")
            st.caption("Optimizasyon yapılmadan önceki ham model performansları:")
            if 'df_treatment_results' not in st.session_state:
                df_treatment_results = ml.ml_cross_validation(X_train, y_train_treatment)
                st.session_state['df_treatment_results'] = df_treatment_results
            else:
                df_treatment_results = st.session_state['df_treatment_results']

            st.dataframe(
                df_treatment_results.style.highlight_max(
                    axis=0, color="lightgreen", subset=["F1-Score (Mean)", "ROC-AUC (Mean)"]
                ),
                width="stretch",
            )

            # İsteğe bağlı: En yüksek skorlu modeli vurgulama
            top_treatment_model = df_treatment_results.iloc[0]["Model"]
            top_treatment_f1 = df_treatment_results.iloc[0]["F1-Score (Mean)"]
            st.success(
                f"🏆 **Treatment için En Başarılı Model:** `{top_treatment_model}` (F1-Score: {top_treatment_f1:.4f})"
            )

            # 2. MODEL BÖLÜMÜ: Turnover Risk Tahmini
            print('--- 2. Turnover Risk Modelleri Yarışıyor ---')
            if 'df_turnover_results' not in st.session_state:
                df_turnover_results = ml.ml_cross_validation(X_train, y_train_turnover)
                st.session_state['df_turnover_results'] = df_turnover_results
            else:
                df_turnover_results = st.session_state['df_turnover_results']

            st.dataframe(
                df_turnover_results.style.highlight_max(
                    axis=0, color="lightgreen", subset=["F1-Score (Mean)", "ROC-AUC (Mean)"]
                ),
                width="stretch",
            )

            top_turnover_model = df_turnover_results.iloc[0]["Model"]
            top_turnover_f1 = df_turnover_results.iloc[0]["F1-Score (Mean)"]
            st.success(
                f"🏆 **Turnover risk için En Başarılı Model:** `{top_turnover_model}` (F1-Score: {top_turnover_f1:.4f})"
            )

            st.subheader("2️⃣ Optuna Hiperparametre Optimizasyonu")
            
            if st.button("🎯 Optuna İle Best Parametreleri Hesapla", width="stretch"):
                with st.spinner("Optuna her model için ayrı parametre kümelerini arıyor..."):
                    if "best_params_treatment_dict" not in st.session_state and "best_params_turnover_dict" not in st.session_state:
                        best_params_treatment_dict = ml.optimize_all_models_with_optuna(X_train, y_train_treatment, n_trials=15)
                        best_params_turnover_dict = ml.optimize_all_models_with_optuna(X_train, y_train_turnover, n_trials=15)
                        st.session_state["best_params_treatment_dict"] = best_params_treatment_dict
                        st.session_state["best_params_turnover_dict"] = best_params_turnover_dict
                        st.success("✅ Optuna hesaplaması tamamlandı!")
                    else:
                        best_params_treatment_dict = st.session_state["best_params_treatment_dict"]
                        best_params_turnover_dict = st.session_state["best_params_turnover_dict"]

            if "best_params_treatment_dict" in st.session_state and "best_params_turnover_dict" in st.session_state:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.write("📋 **Treatment Modellere Özel Best Parametreler:**")
                    st.json(st.session_state["best_params_treatment_dict"])
                with col_p2:
                    st.write("📋 **Turnover Risk Modellere Özel Best Parametreler:**")
                    st.json(st.session_state["best_params_turnover_dict"])

                st.divider()

            # ----------------------------------------------------------------------
            # 3. BEST PARAMETRELER İLE MODELLERİ KARŞILAŞTIRMA & OVERFİT TABLOSU
            # ----------------------------------------------------------------------
            st.subheader("3️⃣ Best Parametrelerle Modelleri Karşılaştır & Overfit Riski")

            if st.button("🚀 Best Parametrelerle Tüm Modelleri Karşılaştır", width="stretch", type="primary"):
                with st.spinner("Modeller yeni parametrelerle tekrar test ediliyor ve overfit oranları hesaplanıyor..."):
                    if "df_tuned_treatment" not in st.session_state and "df_tuned_turnover" not in st.session_state:
                        df_tuned_treatment = ml.evaluate_all_tuned_models(
                            X_train, y_train_treatment, st.session_state["best_params_treatment_dict"]
                        )
                        df_tuned_turnover = ml.evaluate_all_tuned_models(
                            X_train, y_train_turnover, st.session_state["best_params_turnover_dict"]
                        )
                        st.session_state["df_tuned_treatment"] = df_tuned_treatment
                        st.session_state["df_tuned_turnover"] = df_tuned_turnover
                    else: 
                        df_tuned_treatment = st.session_state["df_tuned_treatment"]
                        df_tuned_turnover = st.session_state["df_tuned_turnover"]

            if "df_tuned_treatment" in st.session_state:
                df_tuned_treatment = st.session_state["df_tuned_treatment"]
                df_tuned_turnover = st.session_state["df_tuned_turnover"]

                # --- Treatment Modelleri (Üst Tablo) ---
                st.markdown("### 🎯 **Treatment Tuned Model Sonuçları**")
                st.dataframe(
                    df_tuned_treatment.style.highlight_max(axis=0, color="lightgreen", subset=["Tuned CV F1 (Val)", "Tuned CV ROC-AUC (Val)"]),
                    width="stretch"
                )
                
                # 🛠️ [EKLENDİ]: Treatment Model Seçim Kutusu & Session'a Atma
                treatment_model_list = df_tuned_treatment["Model"].tolist()
                selected_treatment = st.selectbox(
                    "📌 Kullanmak istediğiniz Treatment Modelini Seçin:",
                    options=treatment_model_list,
                    index=0,  # Otomatik olarak en yüksek F1 alan 1. model seçili gelir
                    key="sb_selected_treatment_model"
                )
                st.session_state["selected_treatment_model"] = selected_treatment

                # Ekrandaki bilgi kutusunu seçilen modele göre dinamik yapıyoruz
                chosen_t_row = df_tuned_treatment[df_tuned_treatment["Model"] == selected_treatment].iloc[0]
                st.success(f"🏆 Seçilen Treatment Modeli: `{chosen_t_row['Model']}` (F1: {chosen_t_row['Tuned CV F1 (Val)']:.4f} | Overfit: {chosen_t_row['Overfit Riski']})")

                st.divider()

                # --- Turnover Risk Modelleri (Alt Tablo) ---
                st.markdown("### 🎯 **Turnover Risk Tuned Model Sonuçları**")
                st.dataframe(
                    df_tuned_turnover.style.highlight_max(axis=0, color="lightgreen", subset=["Tuned CV F1 (Val)", "Tuned CV ROC-AUC (Val)"]),
                    width="stretch"
                )
                
                # 🛠️ [EKLENDİ]: Turnover Model Seçim Kutusu & Session'a Atma
                turnover_model_list = df_tuned_turnover["Model"].tolist()
                selected_turnover = st.selectbox(
                    "📌 Kullanmak istediğiniz Turnover Risk Modelini Seçin:",
                    options=turnover_model_list,
                    index=0,  # Otomatik olarak en yüksek F1 alan 1. model seçili gelir
                    key="sb_selected_turnover_model"
                )
                st.session_state["selected_turnover_model"] = selected_turnover

                # Ekrandaki bilgi kutusunu seçilen modele göre dinamik yapıyoruz
                chosen_turn_row = df_tuned_turnover[df_tuned_turnover["Model"] == selected_turnover].iloc[0]
                st.success(f"🏆 Seçilen Turnover Risk Modeli: `{chosen_turn_row['Model']}` (F1: {chosen_turn_row['Tuned CV F1 (Val)']:.4f} | Overfit: {chosen_turn_row['Overfit Riski']})")
            
            st.session_state["X_train"] = X_train
            
        else:
            st.write('**Train verisinin olduğu csv file okunamadı veya böyle bir dosya yok:**')

    elif st.session_state.sub_menu == "SHAP & Bias Analizi":
        ########### SHAP & BIAS ANALIZI ##################
        st.subheader('🔍 SHAP Analizi & Bias (Gender/Country) Kontrolü')
        if "X_train" not in st.session_state:
            st.error("❌ HATA: Train verisi bulunamadı!")
            st.warning("Lütfen önce **Çalışan Analizi > Modelleme** adımına giderek analizi çalıştırın. 'shap_data_ready' doğrulanmadan bu rapor üretilemez.")
            st.stop()
        else:
            X_train = st.session_state["X_train"]
        

        if "selected_treatment_model" in st.session_state and "best_params_treatment_dict" in st.session_state and "selected_turnover_model" in st.session_state and "best_params_turnover_dict" in st.session_state:
            print(f"SHAP Analizi için seçilen Treatment Modeli: {st.session_state['selected_treatment_model']}")
            print(f"SHAP Analizi için seçilen Turnover Modeli: {st.session_state['selected_turnover_model']}")

            y_train_treatment = st.session_state["y_train_treatment"]
            y_train_turnover = st.session_state["y_train_turnover"]

            final_model_treatment, final_model_turnover = ml.fit_both_final_models(X_train=X_train,session_state=st.session_state)
            st.session_state['final_model_treatment'] = final_model_treatment
            st.session_state['final_model_turnover'] = final_model_turnover
                # SHAP fonksiyonuna fit edilmiş modeli veriyoruz
            explainer_treatment, shap_values_treatment, df_shap_importance_treatment = ml.calculate_shap_and_bias(
                final_model_treatment, 
                X_train
            )
            st.session_state["shap_data_treatment"] = {
                        "explainer": explainer_treatment,
                        "shap_values": shap_values_treatment,
                        "mean_abs_shap": df_shap_importance_treatment
                    }

            # SHAP fonksiyonuna fit edilmiş modeli veriyoruz
            explainer_turnover, shap_values_turnover, df_shap_importance_turnover = ml.calculate_shap_and_bias(
                final_model_turnover, 
                X_train
            )
            st.session_state["shap_data_turnover_risk"] = {
                    "explainer": explainer_turnover,
                    "shap_values": shap_values_turnover,
                    "mean_abs_shap": df_shap_importance_turnover
                }
        
            # 1. En Önemli Değişkenler Tablosu
            st.write('### 1. Treatment için Model Kararlarını En Çok Etkileyen Özellikler')
            st.dataframe(df_shap_importance_treatment.head(10), width='stretch')

            # 2. SHAP Summary Plot (Genel Etki)
            st.write('### 2. SHAP Summary Plot')
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.summary_plot(
                shap_values_treatment[:, :, 1]
                if len(shap_values_treatment.shape) == 3
                else shap_values_treatment,
                X_train,
                plot_type="bar",
                show=False,
            )
            st.pyplot(fig, clear_figure=True)

            # 3. Gender / Country Bias İncelemesi
            st.write('### 3. Cinsiyet ve Ülke (Gender/Country) Bias Denetimi')

            # Gender ve Country kolonlarının SHAP önem sırasındaki yerini bulalım
            gender_rank = (
                df_shap_importance_treatment[
                    df_shap_importance_treatment['Feature'].str.contains('gender', case=False)
                ]
                if 'gender' in str(X_train.columns).lower()
                else None
            )
            country_rank = (
                df_shap_importance_treatment[
                    df_shap_importance_treatment['Feature'].str.contains('country', case=False)
                ]
                if 'country' in str(X_train.columns).lower()
                else None
            )

            col1, col2 = st.columns(2)
            with col1:
                st.info('**Gender Etkisi:**')
                st.dataframe(gender_rank, width='stretch')
            with col2:
                st.info('**Country Etkisi:**')
                st.dataframe(country_rank, width='stretch')

            st.success(
                '💡 **Bias Yorumu:** Eğer Gender veya Country değişkenleri en üst sıralarda değilse, '
                'model kararlarını hassas demografik özelliklerden ziyade iş/psikolojik süreçlere göre veriyor demektir (Adil Model).'
            )
        
            st.session_state["shap_data_ready"] = True


            # 1. En Önemli Değişkenler Tablosu
            st.write('### 1. Turnover risk için için Model Kararlarını En Çok Etkileyen Özellikler')
            st.dataframe(df_shap_importance_turnover.head(10), width='stretch')

            # 2. SHAP Summary Plot (Genel Etki)
            st.write('### 2. SHAP Summary Plot')
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.summary_plot(
                shap_values_turnover[:, :, 1]
                if len(shap_values_turnover.shape) == 3
                else shap_values_turnover,
                X_train,
                plot_type="bar",
                show=False,
            )
            st.pyplot(fig, clear_figure=True)

            # 3. Gender / Country Bias İncelemesi
            st.write('### 3. Cinsiyet ve Ülke (Gender/Country) Bias Denetimi')

            # Gender ve Country kolonlarının SHAP önem sırasındaki yerini bulalım
            gender_rank = (
                df_shap_importance_turnover[
                    df_shap_importance_turnover['Feature'].str.contains('gender', case=False)
                ]
                if 'gender' in str(X_train.columns).lower()
                else None
            )
            country_rank = (
                df_shap_importance_turnover[
                    df_shap_importance_turnover['Feature'].str.contains('country', case=False)
                ]
                if 'country' in str(X_train.columns).lower()
                else None
            )

            col1, col2 = st.columns(2)
            with col1:
                st.info('**Gender Etkisi:**')
                st.dataframe(gender_rank, width='stretch')
            with col2:
                st.info('**Country Etkisi:**')
                st.dataframe(country_rank, width='stretch')

    elif st.session_state.sub_menu == "Model Tahminleme":
        print("Model Tahminleme Menüsü")
        st.subheader("4️⃣ Örnek Çalışan Grubu (10 Kişi) İçin Canlı Risk & Destek Tahmini")
        if "X_train" not in st.session_state:
            st.error("❌ HATA: Train verisi bulunamadı!")
            st.warning("Lütfen önce **Çalışan Analizi > Modelleme** adımına giderek analizi çalıştırın. 'shap_data_ready' doğrulanmadan bu rapor üretilemez.")
            st.stop()
        else:
            X_train = st.session_state["X_train"]
            print(f"X_train shape: {X_train.shape}")

        if "final_model_treatment" in st.session_state:
            print(f"Modelleme Analizi için seçilen Treatment Modeli: {st.session_state['selected_treatment_model']}")
            final_model_treatment = st.session_state["final_model_treatment"]

        if "final_model_turnover" in st.session_state:
            print(f"Modelleme Analizi için seçilen Turnover Modeli: {st.session_state['selected_turnover_model']}")
            final_model_turnover = st.session_state["final_model_turnover"]

        
        if "best_params_treatment_dict" in st.session_state and "best_params_turnover_dict" in st.session_state:
            best_params_treatment = st.session_state["best_params_treatment_dict"]
            best_params_turnover = st.session_state["best_params_turnover_dict"]

        if final_model_treatment is None and final_model_turnover is None:
            final_model_treatment, final_model_turnover = ml.fit_both_final_models(X_train, st.session_state)

        if st.button("🎲 Test Verisinden 10 Çalışan Seç ve Tahmin Et", type="primary", width="stretch"):
            with st.spinner("Çalışan profilleri analiz ediliyor..."):
                # Modüler fonksiyondan 10 kişilik tahminli veriyi çekiyoruz
                df_results = ml.predict_sample_employees(
                    file_path=TEST_DATA_PATH,
                    X_train=X_train,
                    model_treatment=final_model_treatment,
                    model_turnover=final_model_turnover,
                    sample_size=10
                )

                st.markdown("### 📋 10 Çalışanın Model Tahmin Özeti")
                
                # Kullanıcıya sadece en kritik sütunları gösteren özet tablo
                summary_cols = [
                    "Destek_Durumu", "Destek_Alsa_Olasiligi_%", 
                    "Terk_Durumu", "Terk_Etme_Olasiligi_%"
                ]
                
                # Varsa çalışan id/isim kolonu da eklenebilir
                st.dataframe(df_results[summary_cols], width="stretch")

                st.divider()

                # Kart Görünümü (İstersen çalışanları tek tek incelemek için):
                st.markdown("### 👤 Kişi Bazlı Detaylı Kart Görünümü")
                for i, (idx, row) in enumerate(df_results.iterrows(), 1):
                    with st.expander(f"Çalışan #{i} — Destek: {row['Destek_Durumu']} | Terk Riski: {row['Terk_Durumu']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Destek Alır mı?**")
                            st.write(f"Tahmin: **{row['Destek_Durumu']}**")
                            st.write(f"Destek İhtiyaç Olasılığı: **%{row['Destek_Alsa_Olasiligi_%']}**")
                        
                        with col2:
                            st.markdown("**Şirketi Terk Eder mi?**")
                            st.write(f"Tahmin: **{row['Terk_Durumu']}**")
                            st.write(f"Ayrılma (Turnover) Olasılığı: **%{row['Terk_Etme_Olasiligi_%']}**")
elif st.session_state.main_menu == "İş Veren":
    if st.session_state.sub_menu == "LLM Yorumları":
        st.header("🤖 Modül 8: LLM ile Gerçek SHAP Analizi & Yorumlama")

        st.markdown("### ⚙️ LLM Ayarları")
        col_llm1, col_llm2 = st.columns(2)
        
        with col_llm1:
            provider = st.selectbox(
                "LLM Sağlayıcısı Seçin:", 
                ["Gemini", "Ollama (Local)", "Cohere"],
                key="llm_provider_select_unique"
            )
        
        with col_llm2:
            ollama_model = "minimax-m3:cloud"
            if provider == "Ollama (Local)":
                ollama_model = st.text_input(
                    "Local Ollama Model Adı:", 
                    value="minimax-m3:cloud",
                    key="ollama_model_input_unique"
                )

        st.divider()

        if not st.session_state.get("shap_data_ready", False):
            st.error("❌ HATA: SHAP Analiz Verileri Bulunamadı!")
            st.warning("Lütfen önce Çalışan Analizi - Modelleme adımına giderek analizi çalıştırın. 'shap_data_ready' doğrulanmadan bu rapor üretilemez.")
        else:
        
            tab1, tab2 = st.tabs(["📊 Gerçek SHAP -> İK Raporu", "💬 Yorum Duygu & Stres Analizi"])

            # --- TAB 1: GERÇEK SHAP DÖNÜŞTÜRÜCÜ ---
            with tab1:
                st.subheader("Modellemeden Gelen Gerçek SHAP Verilerini Yorumlatma")
                
                target_choice = st.radio(
                    "Yorumlanacak Hedef Değişkeni Seçin:",
                    options=["treatment", "turnover_risk"],
                    format_func=lambda x: "🚨 Treatment (Tedavisi/Destek İhtiyacı)" if x == "treatment" else "📉 Turnover Risk (İşten Ayrılma Riski)"
                )

                selected_key = f"shap_data_{target_choice}"
                
                # shap_ready True olsa bile hedef değişken verisi session_state'te var mı kontrolü
                if selected_key in st.session_state:
                    shap_info = st.session_state[selected_key]
                    df_mean_shap = shap_info["mean_abs_shap"]
                    
                    # LLM için en önemli ilk 10 özelliği sözlük yapısına getiriyoruz
                    top_10_features = df_mean_shap.head(10).to_dict(orient="records")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"### 🔝 {target_choice.upper()} En Önemli 10 Öznitelik (SHAP):")
                        st.dataframe(df_mean_shap.head(10), width="stretch")
                        
                        generate_btn = st.button("🚀 İK Raporunu Üret (LLM)", key="shap_llm_btn", width="stretch")

                    with col2:
                        if generate_btn:
                            with st.spinner(f"{provider} SHAP verilerini analiz ediyor ve İK raporu hazırlıyor..."):
                                hr_report = llm.explain_shap_for_hr(
                                    target_name=target_choice,
                                    employee_or_global_id=f"Genel_{target_choice.upper()}_Modeli",
                                    top_features=top_10_features,
                                    provider=provider,
                                    model_name=ollama_model
                                )
                                st.success("✅ Üretilen İK Raporu:")
                                st.info(hr_report)
                else:
                    st.error(f"Seçilen '{target_choice}' modeline ait detaylı SHAP matrisi session_state içinde bulunamadı. Lütfen o modelin eğitimini tekrarlayın.")

            # --- TAB 2: STRES VE DUYGU ANALİZİ ---
            with tab2:
                st.subheader("Açık Uçlu Çalışan Yorum Analizi (why_or_why_not)")
                
                # 1. Varsayılan metin veya state kontrolü
                default_text = "Son 3 aydır neredeyse her gün mesaiye kalıyorum ve yol çok uzun sürüyor. Yönetimden bu konuda bir destek göremedim, çok tükendiğimi hissediyorum."
                
                if "current_comment" not in st.session_state:
                    st.session_state.current_comment = default_text

                # 2. Random yorum seçme butonu
                if st.button("🎲 Veri Setinden Rastgele Yorum Getir", use_container_width=True):
                    # NaN olmayan / boş olmayan yorumları filtreliyoruz
                    valid_comments = df_osmi["comments"].dropna()
                    valid_comments = valid_comments[valid_comments.astype(str).str.strip() != ""]
                    
                    if not valid_comments.empty:
                        st.session_state.current_comment = valid_comments.sample(1).values[0]
                        st.rerun()
                    else:
                        st.warning("Veri setinde analiz edilecek geçerli yorum bulunamadı!")

                # 3. Yorum Alanı (value -> session_state'e bağlı)
                user_comment = st.text_area(
                    "Analiz Edilecek Çalışan Yorumu:", 
                    value=st.session_state.current_comment, 
                    height=120
                )
                
                # 4. Analiz Butonu (width='stretch' yerine use_container_width=True)
                if st.button("🔍 Yorumu Analiz Et", key="comment_btn", use_container_width=True):
                    with st.spinner(f"{provider} duygu ve stres analizini gerçekleştiriyor..."):
                        res = llm.analyze_comment_sentiment_and_stress(
                            comment_text=user_comment,
                            provider=provider,
                            model_name=ollama_model
                        )
                        
                        st.divider()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Duygu Durumu", res.get("sentiment", "N/A"))
                        c2.metric("Stres Seviyesi", f"{res.get('stress_level', '-')} / 5")
                        c3.metric("Kök Neden", res.get("root_cause", "N/A"))
                        
                        st.markdown(f"**Özet:** {res.get('summary', '')}")
                        st.markdown(f"**Aksiyon Planı:** {res.get('action_plan', '')}")

    elif st.session_state.sub_menu == "Öneriler":
        st.session_state["llm_run"] = False
        if not st.session_state.get("kmeans_ready", False) or st.session_state.get("df_with_clusters") is None:
            st.error("❌ HATA: K-Means Gruplama Sonuçları Bulunamadı! Lütfen önce Çalışan Analizi - K-Means Kümeleme adımına giderek analizi çalıştırın. 'kmeans_ready' doğrulanmadan bu rapor üretilemez.")
        else:
            df_clusters = st.session_state.df_with_clusters
            df_spot_data = st.session_state.get("df_spot", None)

            st.header("🎯 Persona Bazlı LLM Çözüm ve Spotify Reçete Motoru")
            print("🎯 RERUN için log amaçlı eklendi")
            # --- LLM PARAMETRELERİ ---
            col_llm1, col_llm2 = st.columns(2)
            with col_llm1:
                provider = st.selectbox("Hedef LLM Sağlayıcısı:", ["Gemini", "Ollama (Local)", "Cohere"], key="mod4_prov")
            with col_llm2:
                ollama_model = "minimax-m3:cloud"
                if provider == "Ollama (Local)":
                    ollama_model = st.text_input("Ollama Model Adı:", value="minimax-m3:cloud", key="mod4_oll")

            # --- MULTISELECT ---
            all_clusters = df_clusters['Cluster_Name'].unique().tolist()
            selected_clusters = st.multiselect(
                "Analiz edilecek persona gruplarını seçin:",
                options=all_clusters,
                default=all_clusters
            )


            # --- BURASI ÇOK KRİTİK! DOĞRU FONKSİYON ÇAĞRILIYOR MU? ---
            if st.button("🚀 Her Persona İçin Ayrı LLM Analizi Başlat", width="stretch"):
                if not selected_clusters:
                    st.warning("Lütfen en az 1 grup seçin.")
                else:
                    with st.spinner("Personalar tek tek LLM'e gönderiliyor..."):
                        # GERÇEK DÖNGÜSEL ÇAĞRI BURADA OLMALI:
                        results = llm.process_all_selected_personas(
                            selected_clusters=selected_clusters,
                            df_clusters=df_clusters,
                            df_spot=df_spot_data,
                            provider=provider,
                            model_name=ollama_model
                        )
                        print("🎯 RERUN için log amaçlı eklendi ---- LLM Porcess")

                        st.session_state["persona_llm_results"] = results
                    st.success("✨ Tüm Personalar LLM Tarafından Ayrı Ayrı Yorumlandı!")


                    print("🎯 RERUN için log amaçlı eklendi ---- 222")
                    # ============================================================
                    # PERSONA RAPORLARI
                    # ============================================================

                    if "persona_llm_results" in st.session_state:
                        results = st.session_state.get("persona_llm_results", {})
                        st.session_state["llm_run"] = True

                        print("🎯 RERUN için log amaçlı eklendi ---- 333")

                        # Render döngüsü
                        for c_name, data in results.items():
                            with st.expander(f"📊 {c_name.upper()} - Özel LLM Yorum Raporu", expanded=True):
                                m = data["metrics"]
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Gruptaki Çalışan", f"{m['toplam_calisan']} Kişi")
                                c2.metric("Daha Önce Terapi Alanlar", f"%{m.get('gecmiste_veya_simdi_destek_alanlar_yuzdesi', 0)}")
                                c3.metric("Farkındalık / Destek Oranı", f"%{m.get('farkindalik_yardim_isteme_yuzdesi', 0)}")
                                st.write("<br>", unsafe_allow_html=True)
                                aksiyon = data["aksiyon_turu"]

                                is_high_risk = "Terapi" in aksiyon or "Support" in aksiyon

                                if "Terapi" in aksiyon or "Support" in aksiyon:
                                    st.error(f"🚨 **LLM KARARI:** {aksiyon}")
                                else:
                                    st.success(f"🎵 **LLM KARARI:** {aksiyon} (Önerilen Mood: **{data['secilen_mood']}**)")

                                st.markdown("**💡 İK Stratejik Analiz & Yol Haritası:**")
                                st.info(data["ik_strateji_onerisi"])

                                st.markdown("**✉️ Çalışan Grubuna Samimi Mesaj:**")
                                st.warning(data["calisan_on_yazi"])

                                # Spotify Listeleme
                                tracks = data["spotify_tracks"]
                                print(f"Spotify önerilen içerikler: {tracks}")
                                if tracks:
                                    st.markdown(f"### 🎧 Havuzdan Önerilen İçerikler:")
                                    for idx, track in enumerate(tracks, 1):
                                        st.markdown(f"**{idx}.** 🎵 **[{track['title']}]({track['url']})** - *{track['artist']}*")

                                st.divider()

            
                  
    elif st.session_state.sub_menu == "Aksiyon":
        ###### LLM & N8N Aksiyon Tetikleme Merkezi ######
        ################################################
        if st.session_state["llm_run"] == True and "persona_llm_results" in st.session_state:
            st.subheader("⚡ Aksiyon Tetikleme Merkezi")
            results = st.session_state.get("persona_llm_results", {})

            persona_list = list(results.keys())

            col_select, col_btn = st.columns([3, 1], vertical_alignment="bottom")
            
            with col_select:
                selected_persona = st.selectbox(
                    "Aksiyon alınacak personayı seçin:",
                    options=persona_list,
                    key="sb_selected_persona"
                )
            
            if st.button("🚀 Aksiyonu Tetikle", use_container_width=True, type="primary"):
                print("N8N için IF bloğuna girdi")                    

                p_data = results[selected_persona]
                df_spot = st.session_state.get("df_spot")
                
                with st.spinner(f"'{selected_persona}' için n8n akışı karar veriliyor ve başlatılıyor..."):
                    # TEK METHOD: n8n kendi içinde HR mı Wellness mı karar verir ve atar
                    print("N8N çağırdı")                    
                    success, message = n8n.trigger_persona_action(selected_persona, p_data, df_spot)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)  
        else:
            st.warning("⚠️ Lütfen önce **Öneriler** sekmesinde LLM analizini çalıştırın ve aksiyon önerilerini alın. Aksiyon tetikleme için LLM sonuçları gereklidir.")     
else:
    st.markdown(f"### 🚀 {st.session_state.main_menu} Modülü")
    st.info(f"Geliştirme aşamasında: {st.session_state.main_menu} içerikleri yakında eklenecek.")