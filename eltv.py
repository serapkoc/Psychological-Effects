from matplotlib import patches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def calculate_eltv(df):
    """ELTV (Çalışan Bağlılık ve Kalma Ömrü Skoru) Hesaplayıcı

    En sade ve anlaşılır versiyon.
    """
    df_eltv = df.copy()

    # ADIM 1: Metin Yanıtlarını Sayısal Puanlara Çevirme Haritası
    leave_puan_haritasi = {
        'Very easy': 100,
        'Somewhat easy': 75,
        "Don't know": 50,
        'Neither easy nor difficult': 50,
        'Somewhat difficult': 25,
        'Very difficult': 0,
        'Difficult': 0,
    }

    work_puan_haritasi = {
        'Never': 100,
        'Rarely': 75,
        'Sometimes': 50,
        'Often': 25,
    }

    turnover_puan_haritasi = {
        0: 100,  # Düşük risk -> Yüksek kalıcılık puanı
        1: 50,  # Orta risk -> Nötr puan
        2: 0,  # Yüksek risk -> Düşük kalıcılık puanı
    }

    # 3. Puan Dönüşümleri
    leave_puan = df_eltv['leave'].map(leave_puan_haritasi).fillna(50)
    work_puan = df_eltv['work_interfere'].map(work_puan_haritasi).fillna(50)

    if 'turnover_risk' in df_eltv.columns:
        retention_puan = (
            df_eltv['turnover_risk'].map(turnover_puan_haritasi).fillna(50)
        )
    else:
        retention_puan = 50

    # 4. GÜÇLENDİRİLMİŞ NİHAİ ELTV FORMÜLÜ
    # %40 Ayrılmama İhtimali + %30 İş Etkilenmeme + %30 İzin Esnekliği
    df_eltv['ELTV_Score'] = (
        (retention_puan * 0.40) + (work_puan * 0.30) + (leave_puan * 0.30)
    )

    # ADIM 4: Skorlara Göre Çalışan Segmenti Oluşturma
    kosullar = [
        (df_eltv['ELTV_Score'] >= 75),  # 75 - 100
        (df_eltv['ELTV_Score'] >= 50) & (df_eltv['ELTV_Score'] < 75),  # 50 - 74
        (df_eltv['ELTV_Score'] >= 25) & (df_eltv['ELTV_Score'] < 50),  # 25 - 49
        (df_eltv['ELTV_Score'] < 25),  # 0 - 24
    ]

    kategoriler = [
        'Yüksek Bağlılık (High)',
        'Orta Bağlılık (Medium)',
        'Düşük Bağlılık (Low)',
        'Kritik / Riskli (Critical)',
    ]

    df_eltv['ELTV_Segment'] = np.select(
        kosullar, kategoriler, default='Bilinmiyor'
    )

    return df_eltv

def plot_eltv_4_segment_matrix(df):
    """4 Segmentli ELTV Görselleştirmesini 5x5 RFM Mantığında Çizen Grafik."""
    df_plot = df.copy()
    segment_counts = df_plot['ELTV_Segment'].value_counts()
    total_count = len(df_plot)

    # 4 Segmentin Matristeki Alanları, İsimleri ve Renkleri
    segment_grid = [
        {
            'key': 'Kritik / Riskli (Critical)',
            'title': 'Kritik / Riskli\n(0-25 Puan)',
            'x': (1, 2),  # Sol Alt Alan (Kırmızı/Turuncu)
            'y': (1, 2),
            'color': '#D9531E',
        },
        {
            'key': 'Düşük Bağlılık (Low)',
            'title': 'Düşük Bağlılık\n(25-50 Puan)',
            'x': (1, 3),  # Orta-Sol Alan (Sarı/Turuncu)
            'y': (3, 3),
            'color': '#F3C148',
        },
        {
            'key': 'Orta Bağlılık (Medium)',
            'title': 'Orta Bağlılık\n(50-75 Puan)',
            'x': (3, 4),  # Orta-Sağ Alan (Açık Yeşil)
            'y': (4, 4),
            'color': '#27AE60',
        },
        {
            'key': 'Yüksek Bağlılık (High)',
            'title': 'Yüksek Bağlılık\n(75-100 Puan)',
            'x': (4, 5),  # Sağ Üst Alan (Koyu Yeşil)
            'y': (5, 5),
            'color': '#2ECC71',
        },
    ]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)

    # Arka plana nötr grid kareleri
    for i in range(1, 6):
        for j in range(1, 6):
            rect = patches.Rectangle(
                (i - 0.5, j - 0.5),
                1,
                1,
                linewidth=0.5,
                edgecolor='white',
                facecolor='#F2F4F4',
            )
            ax.add_patch(rect)

    # Segment Bloklarını Çizdirme
    for seg in segment_grid:
        x1, x2 = seg['x']
        y1, y2 = seg['y']
        width = x2 - x1 + 1
        height = y2 - y1 + 1

        count = segment_counts.get(seg['key'], 0)
        pct = (count / total_count) * 100 if total_count > 0 else 0

        # Bloğu yerleştir
        rect = patches.Rectangle(
            (x1 - 0.5, y1 - 0.5),
            width,
            height,
            linewidth=1.5,
            edgecolor='white',
            facecolor=seg['color'],
        )
        ax.add_patch(rect)

        # Yazıyı ekle
        text_x = x1 - 0.5 + width / 2
        text_y = y1 - 0.5 + height / 2
        ax.text(
            text_x,
            text_y,
            f"{seg['title']}\n{count} Kişi | % {pct:.1f}",
            ha='center',
            va='center',
            color='white',
            fontsize=11,
            fontweight='bold',
        )

    # Eksen Ayrıntıları
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlabel(
        'İzin Esnekliği / Şirket Desteği Skoru', fontsize=11, fontweight='bold'
    )
    ax.set_ylabel(
        'Ruh Sağlığı / İş Etkilenmeme Skoru', fontsize=11, fontweight='bold'
    )
    ax.set_title(
        'ELTV 4-Katmanlı Çalışan Bağlılık Matrisi',
        fontsize=13,
        fontweight='bold',
        pad=15,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig

def get_eltv_kpi_summary(df_eltv):
    """ELTV segmentlerinin sayılarını ve yüzdelerini kartlar (KPI) için hazırlar.

    Dönüş: Dict (Her segment için count ve percentage)
    """
    total = len(df_eltv)
    counts = df_eltv['ELTV_Segment'].value_counts()

    # 4 ana segment için güvenli okuma (Eğer veri setinde o segmentten hiç yoksa 0 döner)
    categories = [
        'Yüksek Bağlılık (High)',
        'Orta Bağlılık (Medium)',
        'Düşük Bağlılık (Low)',
        'Kritik / Riskli (Critical)',
    ]

    summary = {}
    for cat in categories:
        count = counts.get(cat, 0)
        pct = (count / total * 100) if total > 0 else 0
        summary[cat] = {'count': count, 'pct': pct}

    return summary

import numpy as np
import pandas as pd


def calculate_risk_and_stigma_indices(df):
    """3. HAFTA - MODÜL 4: Burnout Risk Skoru ve Stigma Korkusu Endeksi (0-100)

    Hesaplar ve azalan sırada sıralanmış olarak döndürür.
    """
    df_risk = df.copy()

    # --- 1. BURNOUT RISK SKORU MAPLERI ---
    turnover_burnout_map = {2: 100, 1: 50, 0: 0}
    work_burnout_map = {
        'Often': 100,
        'Sometimes': 66,
        'Rarely': 33,
        'Never': 0,
    }
    leave_burnout_map = {
        'Very difficult': 100,
        'Difficult': 100,
        'Somewhat difficult': 75,
        "Don't know": 50,
        'Neither easy nor difficult': 50,
        'Somewhat easy': 25,
        'Very easy': 0,
    }

    turnover_puan = (
        df_risk['turnover_risk'].map(turnover_burnout_map).fillna(50)
        if 'turnover_risk' in df_risk.columns
        else 50
    )
    work_puan = df_risk['work_interfere'].map(work_burnout_map).fillna(50)
    leave_puan = df_risk['leave'].map(leave_burnout_map).fillna(50)

    # NİHAİ BURNOUT RISK SKORU (0-100)
    df_risk['Burnout_Risk_Score'] = (
        (turnover_puan * 0.40) + (work_puan * 0.40) + (leave_puan * 0.20)
    )

    # --- 2. STIGMA KORKUSU ENDEKSİ MAPLERI ---
    # Yanıt olumsuz/korku doluysa puan 100'e yaklaşır
    consequence_map = {'Yes': 100, 'Maybe': 50, 'No': 0}
    coworker_map = {'No': 100, 'Some of them': 50, 'Yes': 0}
    phys_vs_mental_map = {'No': 100, "Don't know": 50, 'Yes': 0}

    consequence_puan = (
        df_risk['mental_health_consequence'].map(consequence_map).fillna(50)
        if 'mental_health_consequence' in df_risk.columns
        else 50
    )
    coworker_puan = (
        df_risk['coworkers'].map(coworker_map).fillna(50)
        if 'coworkers' in df_risk.columns
        else 50
    )
    phys_puan = (
        df_risk['mental_vs_physical'].map(phys_vs_mental_map).fillna(50)
        if 'mental_vs_physical' in df_risk.columns
        else 50
    )

    # NİHAİ STIGMA KORKUSU ENDEKSİ (0-100)
    df_risk['Stigma_Index'] = (
        (consequence_puan * 0.40) + (coworker_puan * 0.30) + (phys_puan * 0.30)
    )

    # --- 3. AZALAN SIRADA LİSTELEME (Sort By Risk) ---
    df_risk_sorted = df_risk.sort_values(
        by=['Burnout_Risk_Score', 'Stigma_Index'], ascending=[False, False]
    )

    return df_risk_sorted