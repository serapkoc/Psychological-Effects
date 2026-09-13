import requests
import streamlit as st


N8N_OSMI_ALERT_WEBHOOK_URL = "http://localhost:5678/webhook-test/osmi-alert"


def trigger_persona_action(selected_persona: str, persona_data: dict, df_spot=None) -> tuple[bool, str]:
    """
    Personanın risk durumunu otomatik analiz eder, ilgili n8n webhook'unu tetikler.
    Returns: (success: bool, message: str)
    """
    is_high_risk = persona_data.get("is_high_risk", False)
    metrics = persona_data.get("metrics", {})
    target_count = metrics.get('toplam_calisan', 0)
    
    if is_high_risk:
        # 🚨 HIGH RISK -> İK ERKEN UYARI AKIŞI
        payload = {
            "risk_group": selected_persona,
            "action_plan": persona_data.get("ik_strateji_onerisi"),
            "target_count": target_count,
            "hr_note": persona_data.get("calisan_on_yazi"),
            "ilave_aksiyon": f"Karar Türü: {persona_data.get('aksiyon', 'Risk Yönetimi')}",
            "target_email": "hr@company.com"
        }
        url = N8N_OSMI_ALERT_WEBHOOK_URL
        action_name = "İK Erken Uyarı Akışı"

    else:
        # 🚀 LOW/MEDIUM RISK -> CUMA ESENLİK BÜLTENİ AKIŞI
        media_link = None
        if df_spot is not None and not df_spot.empty:
            sample_row = df_spot.sample(n=1).iloc[0]
            media_link = sample_row.get("url") or sample_row.get("link") or sample_row.get("youtube_url")

        payload = {
            "risk_group": selected_persona,
            "action_type": f"Mood ({persona_data.get('secilen_mood', 'Müzik')})",
            "target_count": target_count,
            "media_link": media_link or "#",
            "motivation_message": persona_data.get("calisan_on_yazi"),
            "ilave_aksiyon": persona_data.get("ik_strateji_onerisi"),
            "target_email": "employee@company.com"
        }
        url = N8N_OSMI_ALERT_WEBHOOK_URL
        action_name = "Esenlik Bülteni"

    # --- HTTP POST İSTEĞİ ---
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            msg = f"✅ **{selected_persona}** için {action_name} n8n'e başarıyla fırlatıldı!"
            print(msg)
            return True, msg
        else:
            msg = f"❌ n8n Hata ({response.status_code}): {response.text}"
            print(msg)
            return False, msg
    except Exception as e:
        msg = f"🚨 n8n Bağlantı Hatası: {e}"
        print(msg)
        return False, msg