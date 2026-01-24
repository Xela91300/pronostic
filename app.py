import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────────────
ODDS_API_KEY = "b9250f9ec1510f4136bdaca0b1f4f5cf"  # Ta clé The Odds API
ODDS_BASE = "https://api.the-odds-api.com/v4"

st.set_page_config(page_title="Pronos IA + Cotes Réelles", layout="wide")
st.title("Pronostics IA + Cotes Bookmakers en Temps Réel ⚽")
st.caption(f"Mis à jour : {datetime.now().strftime('%H:%M:%S')} | Refresh auto")

refresh_sec = st.sidebar.slider("Rafraîchissement (s)", 60, 300, 120)  # 120s recommandé (quota)
regions = st.sidebar.selectbox("Région bookmakers", ["eu", "uk", "us"], index=0)
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 5, 20, 8)

placeholder = st.empty()

# ─── RÉCUP COTES via The Odds API ──────────────────────────────────────────
@st.cache_data(ttl=refresh_sec - 10)
def fetch_odds():
    url = f"{ODDS_BASE}/sports/soccer/odds/?apiKey={ODDS_API_KEY}&regions={regions}&markets=h2h&oddsFormat=decimal"
    try:
        resp = requests.get(url, timeout=15)
        st.session_state.last_status = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            rows = []
            for event in data:
                home = event['home_team']
                away = event['away_team']
                commence = event['commence_time']
                bookmakers = event.get('bookmakers', [])
                if bookmakers:
                    best_home = min([b['markets'][0]['outcomes'][0]['price'] for b in bookmakers if b['markets']])  # meilleur cote home
                    best_away = min([b['markets'][0]['outcomes'][2]['price'] for b in bookmakers if b['markets']])  # away = index 2 souvent
                    rows.append({
                        "match": f"{home} vs {away}",
                        "ligue": event['sport_key'].replace('soccer_', '').upper(),
                        "commence": commence,
                        "cote_home": best_home,
                        "cote_away": best_away,
                        # Ajoute proba implicite
                        "implied_home": 1 / best_home if best_home > 1 else 0
                    })
            return pd.DataFrame(rows)
        else:
            st.error(f"Erreur API Odds : {resp.status_code} - {resp.text[:200]}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Exception : {str(e)}")
        return pd.DataFrame()

# ─── IA PRONOSTIC SIMPLIFIÉ (ou ton modèle) ────────────────────────────────
def add_ia_proba(df):
    # Simulation ou ton LightGBM ici
    df["proba_home_ia"] = [round(random.uniform(0.45, 0.80), 3) for _ in df.index]
    df["value_home"] = (df["proba_home_ia"] * df["cote_home"]) - 1
    df["value_pct"] = df["value_home"] * 100
    return df

# ─── BOUCLE LIVE ───────────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df_odds = fetch_odds()
        
        if df_odds.empty:
            st.info("Pas de cotes récupérées. Vérifie clé/quota ou essaie plus tard.")
            st.write("Dernier status code :", st.session_state.get('last_status', 'inconnu'))
        else:
            df = add_ia_proba(df_odds)
            
            st.subheader(f"Matchs & Cotes ({len(df)}) – Région : {regions.upper()}")
            
            disp = df[["match", "ligue", "cote_home", "proba_home_ia", "value_pct"]].copy()
            disp["cote_home"] = disp["cote_home"].apply(lambda x: f"{x:.2f}")
            disp["proba_home_ia"] = disp["proba_home_ia"].apply(lambda x: f"{x:.0%}")
            disp["value_pct"] = disp["value_pct"].apply(lambda x: f"+{x:.1f}%" if x > value_threshold else f"{x:.1f}%")
            
            def highlight_value(row):
                if row["value_pct"].startswith("+") and float(row["value_pct"][1:-1]) > value_threshold:
                    return ['background-color: #ccffcc'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                disp.style.apply(highlight_value, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("Value = (proba IA × cote) - 1 → vert si opportunité > seuil")
            st.caption(f"Quota restant : vérifie sur https://the-odds-api.com (usage dans dashboard)")
    
    time.sleep(refresh_sec)
    st.rerun()
