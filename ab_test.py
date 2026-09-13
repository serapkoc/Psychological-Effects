import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

def run_real_ab_test(df, target_col):
    """
    Remote (A) ve Ofis (B) gruplarını sayısal kodlanmış (1 ve 2) 
    hedef değişkenlere göre gerçek bir A/B Testi gibi yarıştırır.
    """
    # remote_work kolonunun Yes/No olduğunu varsayıyoruz (Değilse küçük harfe/temizliğe dikkat)
    clean_df = df[df['remote_work'].isin(['Yes', 'No'])].dropna(subset=[target_col]).copy()
    
    # Grupları ayırıyoruz
    remote_group = clean_df[clean_df['remote_work'] == 'Yes']
    office_group = clean_df[clean_df['remote_work'] == 'No']
    
    # Toplam Örneklem Sayıları (N)
    n_remote = len(remote_group)
    n_office = len(office_group)
    
    # 🎯 SENİN BELİRTTİĞİN SAYISAL RİSK/KRİTER DEĞERLERİ:
    # treatment için risk/hedef değer 1 (Tedavi Gördü)
    # turnover_risk için risk/hedef değer 2 (Ayrılmaya Yatkın)
    positive_value = 1 if target_col == 'treatment' else 2
    
    # Gruplardaki hedef durum sayıları
    success_remote = len(remote_group[remote_group[target_col] == positive_value])
    success_office = len(office_group[office_group[target_col] == positive_value])
    
    # Oranları hesaplayalım (%)
    rate_remote = (success_remote / n_remote) * 100 if n_remote > 0 else 0
    rate_office = (success_office / n_office) * 100 if n_office > 0 else 0
    
    # İKİ ÖRNEKLEM Z-TESTİ
    z_stat, p_value = 0.0, 1.0
    if n_remote > 0 and n_office > 0 and (success_remote + success_office) > 0:
        counts = np.array([success_remote, success_office])
        nobs = np.array([n_remote, n_office])
        z_stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')
    
    # 🏆 KAZANAN VARYANT ANALİZİ
    is_significant = p_value < 0.05
    metric_label = "Tedavi Görme (Kategori: 1)" if target_col == 'treatment' else "İşten Ayrılma Eğilimi (Kategori: 2)"
    
    if is_significant:
        if rate_remote < rate_office:
            status_color = "🟢"
            result_headline = "A Varyantı (Remote) Kazandı!"
            result_text = f"**Uzaktan Çalışanların {metric_label} oranı (%{rate_remote:.1f}), Ofis çalışanlarına göre (%{rate_office:.1f}) istatistiksel olarak anlamlı derecede daha DÜŞÜK.** Bu durum hipotezinizi destekliyor; Remote çalışma modeli çalışan bağlılığını ve motivasyonunu olumlu etkiliyor! 🚀"
        else:
            status_color = "🔵"
            result_headline = "B Varyantı (Ofis) Kazandı!"
            result_text = f"**Ofis Çalışanlarının {metric_label} oranı (%{rate_office:.1f}), Remote çalışanlara göre (%{rate_remote:.1f}) istatistiksel olarak anlamlı derecede daha DÜŞÜK.** Yani ofisten çalışmak motivasyon ve kararlılık açısından mevcut verilere göre daha iyi bir sonuç vermiş görünüyor."
    else:
        status_color = "🟡"
        result_headline = "Test Sonucu: Berabere (İstatistiksel Fark Yok)"
        result_text = f"Remote (%{rate_remote:.1f}) ve Ofis (%{rate_office:.1f}) grupları arasında gözlemlenen fark **istatistiksel olarak anlamlı değil** ($p$-value = {p_value:.4f}). Çalışma modellerinin bu metrik üzerindeki etkisi şans faktöründen ayrıştırılamadı."

    # Görselleştirme Tablosu
    plot_data = pd.DataFrame({
        'Varyant (Çalışma Modeli)': ['A Varyantı (Remote)', 'B Varyantı (Ofis)'],
        'Oran (%)': [rate_remote, rate_office],
        'Örneklem Boyutu (N)': [n_remote, n_office]
    })
    
    fig = px.bar(
        plot_data,
        x='Varyant (Çalışma Modeli)',
        y='Oran (%)',
        text=plot_data['Oran (%)'].apply(lambda x: f"{x:.1f}%"),
        title=f"<b>Varyantlar Arası {metric_label} Oran Karşılaştırması</b>",
        template='plotly_dark',
        color='Varyant (Çalışma Modeli)',
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    
    fig.update_layout(
        height=380, paper_bgcolor='#0e1117', plot_bgcolor='#1e293b',
        font=dict(color='#f8fafc'), showlegend=False
    )
    fig.update_traces(textposition='outside', marker_line_color='#0e1117', marker_line_width=1.5)
    
    report_md = f"""
    ### {status_color} {result_headline}
    {result_text}
    
    ---
    **🔬 A/B Testi Metrik Kartları:**
    *   **Metrik Tanımı:** {metric_label}
    *   **Remote Örneklem (N_A):** {n_remote} kişi | **Ofis Örneklem (N_B):** {n_office} kişi
    *   **Z-İstatistiği:** {round(z_stat, 3)}
    *   **Anlamlılık Değeri ($p$-value):** {round(p_value, 4)}
    """
    
    return fig, report_md



def run_flexible_ab_test(df, mode='combined'):
    """
    mode: 'treatment', 'turnover_risk' veya 'combined'
    Seçilen moda göre Remote vs Ofis grupları arasında Z-Testi koşturur.
    """
    clean_df = df[df['remote_work'].isin(['Yes', 'No'])].dropna(subset=['treatment', 'turnover_risk']).copy()
    
    # Moda göre risk/başarı tanımını yapıyoruz
    if mode == 'treatment':
        clean_df['Risk_Metric'] = (clean_df['treatment'] == 1).astype(int)
        metric_label = "Zihinsel Sağlık Tedavisi Oranı (treatment = 1)"
        brief = "Sadece tedavi görme ihtiyacı duyan çalışanların oranları karşılaştırılıyor."
    elif mode == 'turnover_risk':
        clean_df['Risk_Metric'] = (clean_df['turnover_risk'] == 2).astype(int)
        metric_label = "İşten Ayrılma Eğilimi Oranı (turnover_risk = 2)"
        brief = "Sadece şirketten ayrılmaya daha yatkın olan çalışanların oranları karşılaştırılıyor."
    else: # combined
        clean_df['Risk_Metric'] = ((clean_df['treatment'] == 0) & (clean_df['turnover_risk'] == 2)).astype(int)
        metric_label = "Birleşik Risk / Tükenmişlik Oranı (treatment=0 & turnover_risk=2)"
        brief = "Hem tedavi gören hem de işten ayrılmak isteyen, yani en yüksek riskli kitle karşılaştırılıyor."

    # Grupları ayır
    remote_group = clean_df[clean_df['remote_work'] == 'Yes']
    office_group = clean_df[clean_df['remote_work'] == 'No']
    
    n_remote, n_office = len(remote_group), len(office_group)
    success_remote = remote_group['Risk_Metric'].sum()
    success_office = office_group['Risk_Metric'].sum()
    
    rate_remote = (success_remote / n_remote) * 100 if n_remote > 0 else 0
    rate_office = (success_office / n_office) * 100 if n_office > 0 else 0
    
    # Z-Testi
    z_stat, p_value = 0.0, 1.0
    if n_remote > 0 and n_office > 0 and (success_remote + success_office) > 0:
        counts = np.array([success_remote, success_office])
        nobs = np.array([n_remote, n_office])
        z_stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')
    
    is_significant = p_value < 0.05
    
    # Sonuç Yorumlama
    if is_significant:
        if rate_remote < rate_office:
            status_color = "🟢"
            result_headline = "A Varyantı (Remote) Kazandı!"
            result_text = f"Uzaktan çalışanların bu risk oranı (%{rate_remote:.1f}), Ofis çalışanlarına göre (%{rate_office:.1f}) istatistiksel olarak anlamlı derecede daha **DÜŞÜK**. Remote çalışma modeli bu konuda koruyucu bir kalkan görevi görüyor! 🚀"
        else:
            status_color = "🔵"
            result_headline = "B Varyantı (Ofis) Kazandı!"
            result_text = f"Ofis çalışanlarının bu risk oranı (%{rate_office:.1f}), Remote çalışanlara göre (%{rate_remote:.1f}) istatistiksel olarak anlamlı derecede daha **DÜŞÜK**. Ofis ortamı çalışanların bu metriğini daha olumlu etkilemiş."
    else:
        status_color = "🟡"
        result_headline = "Test Sonucu: Berabere (Net Fark Yok)"
        result_text = f"Remote (%{rate_remote:.1f}) ve Ofis (%{rate_office:.1f}) grupları arasında gözlemlenen fark **istatistiksel olarak anlamlı değil** ($p$-value = {p_value:.4f}). Çalışma modellerinin bu metrik üzerindeki etkisi benzer görünüyor."

    # Grafik Çizimi
    plot_data = pd.DataFrame({
        'Varyant': ['A Varyantı (Remote)', 'B Varyantı (Ofis)'],
        'Oran (%)': [rate_remote, rate_office]
    })
    
    # Moda göre renk paletini değiştirelim, sekmeler arası geçişte tatlı dursun
    colors = ['#636EFA', '#EF553B'] if mode != 'combined' else ['#A6611A', '#DFC27D']
    
    fig = px.bar(
        plot_data, x='Varyant', y='Oran (%)',
        text=plot_data['Oran (%)'].apply(lambda x: f"{x:.1f}%"),
        title=f"<b>{metric_label}</b>", template='plotly_dark',
        color='Varyant', color_discrete_sequence=colors
    )
    print(fig)

    fig.update_layout(height=340, paper_bgcolor='#0e1117', plot_bgcolor='#1e293b', font=dict(color='#f8fafc'), showlegend=False)
    fig.update_traces(textposition='outside', marker_line_color='#0e1117', marker_line_width=1.5)
    

    report_md = f"""
    ### {status_color} {result_headline}
    {result_text}
    
    *ℹ️ {brief}*

    
    ---
    **🔬 İstatistiksel Detaylar:**
    *   **Remote (N):** {n_remote} | **Ofis (N):** {n_office}
    *   **Z-İstatistiği:** {round(z_stat, 3)} | **$p$-value:** {round(p_value, 4)}
    """
    
    return fig, report_md

def run_chi2_tests_double(df, targets=['treatment', 'turnover_risk']):
    """
    Belirtilen hedef değişkenlerin (treatment ve turnover_risk) her biri için 
    veri setindeki diğer tüm kategorik değişkenlerle Ki-Kare testi yapar 
    ve sonuçları tek bir tabloda birleştirir.
    """
    # targets listesindeki kolonların veri setinde olup olmadığını kontrol edelim
    available_targets = [t for t in targets if t in df.columns]
    if not available_targets:
        return pd.DataFrame({"Hata": ["Belirtilen hedef değişkenler veri setinde bulunamadı!"]})
        
    chi2_results = []
    
    # Sadece kategorik (object veya category) kolonları seçiyoruz
    categoricals = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categoricals:
        # Eğer incelenen kolon hedeflerden biriyse veya SurveyID (Country) ise pas geç
        if col in targets or col == 'Country':
            continue
            
        row_data = {"Değişken": col}
        
        # İki hedef için de sırayla Ki-Kare testini koşturuyoruz
        for target in available_targets:
            clean_df = df[[col, target]].dropna()
            
            # Yeterli kategori çeşitliliği var mı kontrolü
            if clean_df[col].nunique() <= 1 or clean_df[target].nunique() <= 1:
                row_data[f"{target} (p-value)"] = "Yetersiz Veri"
                row_data[f"{target} Durum"] = "Test Edilemedi"
                continue
                
            contingency_table = pd.crosstab(clean_df[col], clean_df[target])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Tabloya eklenecek sütunları dolduruyoruz
            row_data[f"{target} (p-value)"] = round(p, 4)
            row_data[f"{target} Durum"] = "🎯 Anlamlı" if p < 0.05 else "❌ Anlamsız"
            
        chi2_results.append(row_data)
        
    # Sonuçları DataFrame'e çevirip ilk hedefin p-değerine göre sıralayalım
    res_df = pd.DataFrame(chi2_results)
    
    # Sıralama sütununu dinamik seçelim (tabloda hangisi varsa)
    sort_col = f"{available_targets[0]} (p-value)" if available_targets else "Değişken"
    return res_df.sort_values(by=sort_col)

def run_chi2_year_treatment(df):
    """
    Veri setindeki 'year' ve 'treatment' kolonlarını kullanarak
    frekans tablosunu (crosstab) oluşturur ve Ki-Kare testini koşturur.
    """
    # Excel'deki pivot tablo gibi, yıllara göre 0 ve 1'lerin gerçek sayılarını çıkarır
    contingency_table = pd.crosstab(df['SurveyID'], df['treatment'])
    
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return {
        "table": contingency_table,
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

###################################################################
#################### **DUZELTME KODLARI** #########################
###################################################################
def change_treatment_type(df):

    print("🔄 Orijinal 'treatment' benzersiz değerleri:", df['treatment'].unique())
    print("🔄 Orijinal Veri Tipi:", df['treatment'].dtype)
    print("🔄 Toplam boş kayıt:", df['treatment'].isnull().sum())


    # 2. Değerleri string fonksiyonlarıyla temizle (boşlukları uçur, küçük harfe çek)
    df['treatment'] = df['treatment'].astype(str).str.strip().str.lower()

    # 3. Olası tüm eşleşmeleri (Yes/No, 1/0, True/False) tam sayı karşılıklarına eşle
    mapping = {
        'yes': 1, '1': 1, '1.0': 1, 'true': 1,
        'no': 0, '0': 0, '0.0': 0, 'false': 0
    }

    # Eşlemeyi yap, eğer haritada olmayan tuhaf bir değer varsa (NaN vs) onu da 0'a çekelim veya temizleyelim
    df['treatment'] = df['treatment'].map(mapping).fillna(0).astype(int)

    # 4. Dosyayı kalıcı olarak üzerine yaz
    df.to_csv("/Users/serapkoc/Desktop/FuFighters/Psychological Effects/dataset/mental_health.csv", index=False)

    print("\n🎯 --- İŞLEM TAMAMLANDI ---")
    print("✅ Güncel 'treatment' benzersiz değerleri:", df['treatment'].unique())
    print("✅ Güncel Veri Tipi:", df['treatment'].dtype)
    print(f"💾 mental_health.csv dosyası başarıyla güncellendi ve kalıcı olarak kaydedildi!")