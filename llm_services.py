import os
import json
import random
from dotenv import load_dotenv
from google import genai
import cohere
import ollama
import requests

# env yükleme
current_dir = os.path.dirname(os.path.abspath(__file__))
for env_name in ["secret.env", "secrets.env"]:
    env_path = os.path.join(current_dir, env_name)
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
        break

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")


def explain_shap_for_hr(
    target_name: str, # 'treatment' veya 'turnover_risk'
    employee_or_global_id: str, 
    top_features: list[dict], 
    provider: str = "Gemini", 
    model_name: str = "llama3"
) -> str:
    """
    Spesifik hedef değişken (treatment veya turnover_risk) için 
    gerçek SHAP öznitelik katkılarını İK diline çevirir.
    """
    
    # Hedef değişkene göre İK bağlamını belirliyoruz
    if target_name == "treatment":
        context_desc = "Çalışanın Psikolojik Destek/Tedavi (Treatment) İhtiyacı Durumu"
    elif target_name == "turnover_risk":
        context_desc = "Çalışanın İşten Ayrılma Riski (Turnover Risk) Durumu"
    else:
        context_desc = f"Hedef Değişken: {target_name}"

    prompt = f"""
    Sen kıdemli bir İnsan Kaynakları (İK) Analitiği Danışmanısın.
    Aşağıda '{employee_or_global_id}' için eğitilen makine öğrenmesi modelinden elde edilen GERÇEK SHAP verileri yer almaktadır.

    ANALİZ EDİLEN HEDEF DEĞİŞKEN: {target_name.upper()} ({context_desc})

    SHAP Faktörleri ve Katkı Düzeyleri:
    {json.dumps(top_features, ensure_ascii=False, indent=2)}

    GÖREVİN:
    Bu teknik verileri doğrudan '{target_name}' hedefini göz önüne alarak profesyonel, yapıcı ve aksiyon odaklı bir İK raporu metnine dönüştür.
    
    KURALLAR:
    1. İstatistiksel/teknik terimler (SHAP, TreeExplainer, Mean_SHAP, XGBoost vb.) KESİNLİKLE KULLANMA.
    2. Hedef değişkene ({target_name}) doğrudan etki eden ilk 3 faktörü vurgulayarak iş hayatı diliyle açıkla.
    3. İK yöneticisi için '{target_name}' durumunu iyileştirecek somut ve uygulanabilir 2 aksiyon önerisi sun.
    4. Maksimum 150-200 kelime olsun. Metin net ve Türkçe yazılmalı.
    """

    if provider == "Gemini":
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        return response.text

    elif provider == "Ollama (Local)":
        response = ollama.chat(
            model="minimax-m3:cloud",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

    elif provider == "Cohere":
        co = cohere.ClientV2(api_key=COHERE_API_KEY)
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content[0].text

    else:
        raise ValueError("Geçersiz LLM Sağlayıcısı!")



# ==========================================
# 2. YORUM DUYGU VE STRES ANALİZİ
# ==========================================
def analyze_comment_sentiment_and_stress(comment_text: str, provider: str = "Gemini", model_name: str = "minimax-m3:cloud") -> dict:
    """
    Çalışanın 'why_or_why_not' açık uçlu yorumunu analiz eder ve JSON formatında döndürür.
    """

    if not comment_text or str(comment_text).strip() == "":
        return {
            "sentiment": "Nötr",
            "stress_level": 1,
            "root_cause": "Yorum yapılmamış",
            "summary": "Çalışan herhangi bir açık uçlu geri bildirimde bulunmadı.",
            "action_plan": "Bir aksiyon planı oluşturmak için çalışanla birebir görüşme yapılması önerilir."
        }

    prompt = f"""
    Aşağıdaki çalışan yorumunu analiz et ve YALNIZCA geçerli bir JSON formatında yanıt ver. Başka hiçbir açıklama ekleme.

    Çalışan Yorumu:
    "{comment_text}"

    Çıktı JSON Şeması:
    {{
        "sentiment": "Pozitif" veya "Nötr" veya "Negatif",
        "stress_level": 1 ile 5 arasında tamsayı (1: Çok Düşük/Yok, 5: Aşırı Yüksek Stres/Tükenmişlik),
        "root_cause": "İş Yükü" veya "Maaş" veya "Yönetim" veya "Yan Haklar" veya "Çalışma Saatleri" veya "Kariyer" veya "Diğer",
        "summary": "Yorumun İK için 1 cümlelik özeti"
        "action_plan": "Çalışanın yorumuna göre önerilen aksiyon planı (3-4 cümle)"

    }}
    """

    raw_response = ""

    if provider == "Gemini":
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        raw_response = response.text

    elif provider == "Ollama (Local)":
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json"  # Ollama JSON zorlaması
        )
        raw_response = response['message']['content']

    elif provider == "Cohere":
        co = cohere.ClientV2(api_key=COHERE_API_KEY)
        response = co.chat(
            model="command-a-03-2025",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        raw_response = response.message.content[0].text

    try:
        return json.loads(raw_response)
    except Exception:
        # JSON parse hatası durumunda fallback
        return {
            "sentiment": "Belirsiz",
            "stress_level": 3,
            "root_cause": "Analiz Edilemedi",
            "summary": raw_response,
            "action_plan": "Aksiyon planı belirlenemedi"
        }


def analyze_single_persona(cluster_name, df_c, df_spot, provider, model_name):
    """
    Tek bir persona grubu için K-Means küme ismini ve metrikleri LLM'e gönderir,
    kural eşiği olmadan küme mantığına göre karar ürettirir.
    """
    dynamic_moods = []
    if df_spot is not None and 'Mood' in df_spot.columns:
        dynamic_moods = df_spot['Mood'].dropna().unique().tolist()

    avg_age = df_c['Age'].mean() if 'Age' in df_c.columns else 30.0
    
    def calc_ratio(col_name):
        if col_name not in df_c.columns:
            return 35.0
        vals = df_c[col_name].astype(str).str.strip().str.lower()
        yes_count = vals.isin(['yes', '1', 'true', 'evet']).sum()
        return (yes_count / len(df_c)) * 100 if len(df_c) > 0 else 0.0

    treatment_ratio = calc_ratio('treatment')
    help_seeking_ratio = calc_ratio('seek_help')

    metrics = {
        "toplam_calisan": int(len(df_c)),
        "yas_ortalamasi": round(float(avg_age), 1),
        "gecmiste_veya_simdi_destek_alanlar_yuzdesi": round(float(treatment_ratio), 1),
        "farkindalik_yardim_isteme_yuzdesi": round(float(help_seeking_ratio), 1)
    }

    # --- KÜME İSMİNE ODAKLI YENİ PROMPT ---
    prompt = f"""
    Sen kıdemli bir Örgütsel Psikolog ve İK Analisti olarak görev yapıyorsun.
    Aşağıda K-Means algoritması ile gruplanmış BİR çalışan persona grubu bilgileri yer almaktadır:
    
    Persona Küme Adı: "{cluster_name}"
    Grup Metrikleri: {json.dumps(metrics, ensure_ascii=False)}
    Mevcut Spotify Mood Etiketleri: {dynamic_moods}
    
    Çok Önemli Karar Kuralları:
    1. Kararını sayısal eşiklere göre değil, "Persona Küme Adı"nın anlamsal içeriğine göre ver:
       - Eğer küme adı 'DÜŞÜK RİSKLİ', 'ORTA RİSKLİ', 'FAKINDALIĞI YÜKSEK' veya 'DESTEK ALANLAR' gibi görece mutlu/stabil/farkındalık sahibi bir grubu ifade ediyorsa (Örn: Küme 0, Küme 1); aksiyon türünü kesinlikle 'Spotify Müzik Reçetesi' seç ve listeden uygun 1 mood belirle.
       - Eğer küme adı 'YÜKSEK RİSKLİ', 'DESTEK ALMAYAN', 'YOĞUN STRES' veya 'KRİZ' içeriyorsa (Örn: Küme 2, Küme 3); aksiyon türünü 'Terapi / Canlı Destek Veya Mentorluk' seç.
    
    2. İK departmanı için bu grubun niteliğine uygun aksiyon stratejisi yaz.
    3. Bu çalışan grubuna gönderilecek, onların durumunu tahlil eden, motive edici ve samimi bir mesaj/ön yazı hazırla.
    
    ÇIKTI FORMATI: Sadece ve sadece geçerli bir JSON döndür. Markdown etiketleri KULLANMA.
    {{
        "aksiyon_turu": "Spotify Müzik Reçetesi VEYA Terapi / Canlı Destek Veya Mentorluk",
        "secilen_mood": "Seçilen mood etiketi VEYA null",
        "ik_strateji_onerisi": "Bu gruba özel detaylı İK stratejisi...",
        "calisan_on_yazi": "Çalışanlara gönderilecek samimi mesaj..."
    }}
    """

    try:
        response_text = generate_generic_response(
            prompt=prompt,
            provider=provider,
            model_name=model_name,
            is_json=True
        )
        clean_json = str(response_text).replace("```json", "").replace("```", "").strip()
        llm_res = json.loads(clean_json)
    except Exception as e:
        # LLM patlaması durumunda da K-Means isim mantığına göre akıllı fallback
        c_lower = cluster_name.lower()
        is_high_risk = "yüksek" in c_lower or "stres" in c_lower or "küme 2" in c_lower or "küme 3" in c_lower
        
        fallback_mood = random.choice(dynamic_moods) if dynamic_moods else "Relaxing"
        llm_res = {
            "aksiyon_turu": "Terapi / Canlı Destek Veya Mentorluk" if is_high_risk else "Spotify Müzik Reçetesi",
            "secilen_mood": None if is_high_risk else fallback_mood,
            "ik_strateji_onerisi": f"{cluster_name} grubu için klinik mentorluk ve iş yükü takibi önerilir." if is_high_risk else f"{cluster_name} grubunun motivasyonunu korumak için dinamik müzik ve sosyal içerikler önerilmiştir.",
            "calisan_on_yazi": f"Merhaba Ekip, zorlu bir dönemde yanınızdayız." if is_high_risk else f"Harika bir hafta geçirmek için size özel ritimler seçtik!"
        }

    # 5. SPOTIFY MATCHING
    matched_tracks = []
    secilen_mood = llm_res.get("secilen_mood")
    
    if llm_res.get("aksiyon_turu") == "Spotify Müzik Reçetesi" and secilen_mood and df_spot is not None:
        if 'Mood' in df_spot.columns:
            df_matched = df_spot[df_spot['Mood'].astype(str).str.lower() == str(secilen_mood).lower()]
            
            if not df_matched.empty:
                sample_count = min(10, len(df_matched))
                df_sampled = df_matched.sample(n=sample_count)
                
                for _, row in df_sampled.iterrows():
                    matched_tracks.append({
                        "title": row.get('Track', row.get('Title', row.get('title', 'Bilinmeyen'))),
                        "artist": row.get('Artist', row.get('artist', 'Bilinmeyen')),
                        "url": row.get('Url', row.get('url', row.get('Link', '#'))),
                        "source": row.get('Source', row.get('source', 'Spotify'))
                    })

    return {
        "cluster_name": cluster_name,
        "metrics": metrics,
        "aksiyon_turu": llm_res.get("aksiyon_turu"),
        "secilen_mood": secilen_mood,
        "ik_strateji_onerisi": llm_res.get("ik_strateji_onerisi"),
        "calisan_on_yazi": llm_res.get("calisan_on_yazi"),
        "spotify_tracks": matched_tracks
    }

def generate_generic_response(prompt, provider, model_name="minimax-m3:cloud", is_json=True):
    """
    Arayüzden seçilen sağlayıcıya (Gemini, Ollama, Cohere) göre 
    istek atıp ham metin/JSON yanıtı döndüren merkezi fonksiyon.
    """
    provider_lower = str(provider).lower()
    print(f"LLM Çağrısı: {provider_lower} - Model: {model_name} - JSON Zorunluluğu: {is_json}")

    # --- A. GEMINI ENTEGRASYONU ---
    if "gemini" in provider_lower:
        try:
            print("gemini içine girdi")
            # JSON çıktısı almak için config ayarlıyoruz
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            #raw_response = response.text
            print(f"LLM SONUÇ: {response.text} ")

            return response.text

        except Exception as e:
            print(f"Gemini API Hatası: {str(e)}")
            raise RuntimeError(f"Gemini API Hatası: {str(e)}")

    # --- B. COHERE ENTEGRASYONU ---
    elif "cohere" in provider_lower:
        try:
            co = cohere.Client(COHERE_API_KEY)
            # JSON formatı zorunluluğu için response_format ekliyoruz
            response = co.chat(
                model="command-a-03-2025",
                messages=[{"role": "user", "content": prompt}]
            )
            print(f"LLM SONUÇ: {response.text} ")
            return response.message.content[0].text

        except Exception as e:
            raise RuntimeError(f"Cohere API Hatası: {str(e)}")

    # --- C. OLLAMA (LOCAL) ENTEGRASYONU ---
    elif "ollama" in provider_lower:
        try:
            response = ollama.chat(
                 model="minimax-m3:cloud",
                messages=[{"role": "user", "content": prompt}])
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama (Local) Bağlantı Hatası: {str(e)}. Sunucunun açık olduğundan emin olun.")

    else:
        raise ValueError(f"Bilinmeyen LLM Sağlayıcısı: {provider}")


    
def process_all_selected_personas(selected_clusters, df_clusters, df_spot, provider, model_name):
    """
    Arayüzden seçilen tüm personaları döngüye sokup her biri için ayrı LLM çağrısı tetikler.
    """
    results = {}
    for c_name in selected_clusters:
        df_c = df_clusters[df_clusters['Cluster_Name'] == c_name]
        # Her persona için ayrı ayrı LLM analizi çalıştırılıyor
        print("LLM çağrısı başlatılıyor...")
        print(f"Processing persona: {c_name} with {len(df_c)} employees...{provider} LLM is being called...{model_name} model is being used...")
        res = analyze_single_persona(c_name, df_c, df_spot, provider, model_name)
        results[c_name] = res
        
    return results