# 🌱 AI-Powered HR Analytics & Employee Well-being Platform

Proje, çalışanların ruh sağlığı, işten ayrılma riski (turnover) ve bağlılık düzeylerini makine öğrenmesi modelleri, istatistiksel testler, LLM (Büyük Dil Modelleri) ve n8n otomasyon akışları ile analiz eden bütüncül bir İnsan Kaynakları analitiği platformudur.

---

🛠️ Proje Mimarisi ve Kod Yapısı
```text
.
├── dataset/
│   ├── mental_health.csv            # Temel işlenmiş veri seti
│   ├── mental_health_train.csv      # Model eğitim veri seti
│   ├── mental_health_test.csv       # Model test veri seti
│   ├── spotify_youtube_new.csv      # Mood & Müzik eşleştirme veri seti
│   ├── hatali_ve_aykiri_raporu.pkl  # Veri kalitesi & Anomali tespit raporu
│   └── fe_summary_file.pkl          # Öznitelik mühendisliği (FE) özet raporu
├── ab_test.py                       # A/B Testi (Z-testi) ve Ki-Kare (Chi-Square) analizleri
├── app.py                           # Ana Streamlit uygulama arayüzü ve navigasyon
├── config.toml                      # Streamlit yapılandırma ayarları
├── eda.py                           # Keşifçi veri analizi, aykırı değer tespiti & Plotly grafikleri
├── eltv.py                          # ELTV (Employee Lifetime Value) ve Tükenmişlik Risk skorlaması
├── feature_engineering.py           # Öznitelik mühendisliği, yeni değişken türetimi & veri ön işleme
├── llm_services.py                  # Multi-LLM (Gemini, Cohere, Ollama) entegrasyonu ve SHAP açıklamaları
├── machine_learning.py              # Makine öğrenmesi modelleri eğitimi, değerlendirmesi ve SHAP hesaplamaları
├── n8n.py                           # n8n Webhook entegrasyonu ve otomatik İK aksiyon tetikleyicileri
├── secrets.env                      # Hassas API anahtarları ve çevre değişkenleri
└── style.css                        # Özel arayüz CSS stilleri

```
## 📌 Öne Çıkan Özellikler ve Modüller

### 1. 📊 Keşifçi Veri Analizi (EDA) ve Anomali Tespiti (`eda.py`)
* **Aykırı Değer Tespiti:** Yaş (`Age`) ve diğer sayısal/kategorik verilerdeki hatalı (mantıksal yaş sınırları dışındaki) veya eksik kayıtlar tespit edilip `hatali_ve_aykiri_raporu.pkl` olarak dondurulur.
* **Karanlık Tema İnteraktif Görseller:** Streamlit paneline uyumlu, Plotly destekli ülke bazlı tedavi oranları ve çapraz frekans analiz grafikleri sunar.

### 2. 🧪 A/B Testleri ve İstatistiksel Analizler (`ab_test.py`)
* **Remote vs. Ofis Karşılaştırması:** Uzaktan çalışanlar ile ofis çalışanlarının tedavi görme (`treatment`) ve işten ayrılma riski (`turnover_risk`) oranları İki Örneklem Z-Testi ile karşılaştırılır.
* **Ki-Kare (Chi-Square) Bağımsızlık Testleri:** Kategorik değişkenler arasındaki istatistiksel anlamlılık ilişkilerini otomatize şekilde raporlar.

### 3. 📈 ELTV (Çalışan Yaşam Boyu Değeri) ve Burnout Analizi (`eltv.py`)
* **ELTV Skoru & 4-Katmanlı Bağlılık Matrisi:** İzin esnekliği, iş etkilenme durumu ve turnover risk puanlarını harmanlayarak çalışanları *Kritik, Düşük, Orta, Yüksek Bağlılık* gruplarına ayırır.
* **Tükenmişlik ve Stigma Endeksi:** Çalışanların psikolojik destek alma korkusu (damgalanma endeksi) ile tükenmişlik risk skorlarını 0-100 arasında hesaplar.

### 4. 🤖 Çoklu LLM Desteği & SHAP Açıklanabilirliği (`llm_services.py`)
* **Çoklu LLM Sağlayıcı Entegrasyonu:** Google Gemini (`gemini-3-flash-preview`), Cohere (`command-a-03-2025`) ve Yerel Ollama (`minimax-m3:cloud`) modelleri dinamik olarak desteklenir.
* **SHAP Yorumlayıcısı:** Karmaşık ML model çıktılarını ve SHAP değerlerini teknik terimlerden arındırarak İK diline uyarlanmış somut aksiyon raporlarına dönüştürür.
* **Açık Uçlu Yorum Analizi:** Çalışan yorumlarından duygu (sentiment), stres seviyesi ve kök neden tespiti yapar.
* **Persona Analizi ve Mood Eşleştirme:** K-Means kümelerini analiz eder; düşük riskli gruplara Spotify müzik reçetesi, yüksek riskli gruplara klinik mentorluk/terapi önerileri sunar.

### 5. ⚡ Otomatik n8n Akış Tetikleme (`n8n.py`)
* **Dinamik Webhook Tetikleme:**
  * 🚨 **Yüksek Riskli Personalar:** İK Erken Uyarı Akışını tetikler ve İK birimine stratejik uyarı e-postası yollar.
  * 🚀 **Düşük/Orta Riskli Personalar:** Çalışan motivasyonunu artırmak için belirlenen Spotify Mood reçetesiyle birlikte "Cuma Esenlik Bülteni" akışını başlatır.

---

## 📄 Python Dosyaları ve İşlev Detayları

### 1. `app.py` — Ana Uygulama & Streamlit Arayüzü
* **Görevi:** Platformun tüm modüllerini tek bir interaktif web panelinde birleştiren ana giriş noktasıdır.
* **Gerçekleştirdiği İşlemler:**
  * Streamlit sayfa düzenini, kenar çubuğu (sidebar) navigasyonunu ve tema yüklemelerini (`style.css` ve `config.toml`) yönetir.
  * EDA, Makine Öğrenmesi, A/B Testi, ELTV, LLM Analizleri ve n8n Otomasyon sekmeleri arasındaki veri akışını koordine eder.
  * Kullanıcıdan alınan parametre ve filtreleri ilgili backend modüllerine ileterek dinamik sonuçlar üretir.

### 2. `eda.py` — Keşifçi Veri Analizi (EDA) & Anomali Tespiti
* **Görevi:** Veri setinin istatistiksel yapısını incelemek ve aykırı/hatalı verileri tespit etmektir.
* **Gerçekleştirdiği İşlemler:**
  * Yaş (`Age`) ve diğer sayısal değişkenlerdeki mantıksal sınırlar dışı verileri ayıklar ve `hatali_ve_aykiri_raporu.pkl` olarak kaydeder.
  * Ülke, cinsiyet, uzaktan çalışma durumu ve tedavi alma oranları arasındaki ilişkileri interaktif Plotly grafikleriyle görselleştirir.
  * Veri kümesindeki eksik değer dağılımlarını ve değişkenler arası korelasyonları analiz eder.

### 3. `feature_engineering.py` — Öznitelik Mühendisliği & Veri Ön İşleme
* **Görevi:** Ham veriyi makine öğrenmesi modellerinin en yüksek başarıyla işleyebileceği formata getirmektir.
* **Gerçekleştirdiği İşlemler:**
  * Eksik verilerin (missing values) uygun stratejilerle (kategorik/sayısal doldurma) işlenmesini sağlar.
  * Kategorik değişkenleri **One-Hot Encoding** veya **Label Encoding** yöntemleriyle dönüştürür.
  * Yeni karma değişkenler (composite features) türeterek çalışanların stres seviyesi, destek alma kolaylığı ve esneklik metriklerini tek bir skora bağlar.
  * İşlem özetlerini ve dönüştürülmüş öznitelik bilgilerini `fe_summary_file.pkl` dosyasına kaydeder.

### 4. `machine_learning.py` — Model Eğitimi & SHAP Açıklanabilirliği
* **Görevi:** Çalışanların ruh sağlığı tedavisi alma ihtiyacı ve işten ayrılma risklerini tahminleyen makine öğrenmesi modellerini eğitmek ve değerlendirmektir.
* **Gerçekleştirdiği İşlemler:**
  * Veri setini `train` ve `test` parçalarına ayırarak Gradient Boosting algoritmaları (CatBoost, LightGBM, XGBoost vb.) üzerinde model eğitimi gerçekleştirir.
  * Accuracy, Precision, Recall, F1-Score ve ROC-AUC metrikleri üzerinden model performanslarını kıyaslar.
  * **SHAP (SHapley Additive exPlanations)** değerlerini hesaplayarak hangi değişkenin tahmine ne yönde katkı sağladığını belirler.

### 5. `ab_test.py` — İstatistiksel Hipotez Testleri & A/B Analizleri
* **Görevi:** Çalışma modelleri ve grup farklarının istatistiksel olarak anlamlı olup olmadığını test etmektir.
* **Gerçekleştirdiği İşlemler:**
  * **İki Örneklem Z-Testi:** Uzaktan çalışanlar (Remote) ile ofis çalışanları arasında tedavi görme oranları veya tükenmişlik riskleri açısından anlamlı bir fark olup olmadığını doğrulukla ölçer.
  * **Ki-Kare (Chi-Square) Bağımsızlık Testi:** Şirket büyüklüğü, esnek çalışma imkanları ve damgalanma (stigma) korkusu gibi kategorik değişkenler arasındaki bağımsızlığı test eder.
  * Test çıktılarının $p$-değerlerini ve güven aralıklarını İK yöneticilerinin rahatça anlayabileceği raporlara dönüştürür.

### 6. `eltv.py` — Çalışan Yaşam Boyu Değeri (ELTV) & Burnout Skorlaması
* **Görevi:** Çalışanların şirket içerisindeki potansiyel kalıcılık değerini (ELTV) ve tükenmişlik/stigma endekslerini hesaplamaktır.
* **Gerçekleştirdiği İşlemler:**
  * İzin alma esnekliği, iş etkilenme durumu ve turnover risk puanlarını harmanlayarak 0-100 arasında **Tükenmişlik Skoru** ve **Stigma Endeksi** üretir.
  * Çalışanları 4 ana katmandan oluşan **Bağlılık Matrisine** (*Kritik, Düşük, Orta, Yüksek Bağlılık*) yerleştirir.
  * K-Means kümeleme yöntemi ile benzer risk grubundaki çalışanları otomatize personalar altında toplar.

### 7. `llm_services.py` — Multi-LLM Entegrasyonu & Yapay Zeka Analizi
* **Görevi:** Sayısal model çıktılarını, SHAP analizlerini ve serbest metin çalışan yorumlarını doğal dile dönüştürerek aksiyon önerileri üretmektir.
* **Gerçekleştirdiği İşlemler:**
  * **Çoklu LLM Desteği:** Google Gemini, Cohere ve yerel Ollama modelleri üzerinden esnek çalışma mimarisi sunar.
  * SHAP analiz çıktılarını teknik terimlerden arındırarak İK uzmanları için yönetici özetine (Executive Summary) dönüştürür.
  * Çalışan yorumlarından duygu analizi (Sentiment Analysis) ve kök-neden tespiti yapar.
  * **Spotify Mood Eşleştirmesi:** Düşük riskli çalışan gruplarına motivasyon artırıcı Spotify çalma listesi önerileri, yüksek riskli gruplara ise klinik mentorluk/destek planları hazırlar.

### 8. `n8n.py` — Otomasyon & Webhook Entegrasyonu
* **Görevi:** Analiz sonuçlarına göre dış sistemleri ve İK bildirim akışlarını otomatik olarak tetiklemektir.
* **Gerçekleştirdiği İşlemler:**
  * Streamlit arayüzünde oluşan sonuçlara göre n8n Webhook uç noktalarına (endpoints) veri gönderir.
  * **Aksiyon Senaryoları:**
    * 🚨 *Yüksek Riskli Çalışanlar:* İK departmanına otomatik erken uyarı e-postası ve müdahale bülteni iletir.
    * 🚀 *Düşük/Orta Riskli Çalışanlar:* Belirlenen Spotify müzik reçetesi ve motivasyon içerikleriyle "Cuma Esenlik Bülteni" akışını başlatır.

---

## ⚙️ Kurulum ve Çalıştırma

1. Depoyu Klonlayın
```bash
git clone [https://github.com/kullanici-adi/repo-adi.git](https://github.com/kullanici-adi/repo-adi.git)
cd repo-adi

2. Gerekli Bağımlılıkları Yükleyin
Bash
pip install -r requirements.txt
3. Ortam Değişkenlerini (Env) Tanımlayın
secrets.env dosyanız içerisine gerekli API anahtarlarınızı ekleyin:

Kod snippet'i
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
N8N_WEBHOOK_URL=your_n8n_webhook_url

4. Streamlit Uygulamasını Başlatın
Bash
streamlit run app.py
