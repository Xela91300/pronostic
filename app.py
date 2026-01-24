import streamlit as st
import pandas as pd
import requests
import time
from datetime import date

# ─── CONFIG ─────────────────────────────────────────────────────────────────
API_KEY = "249b3051eca063f0e381609128c00d7d"  # Ta clé fournie

SPORT_CONFIG = {
    "Football": {
        "base_url": "https://v3.football.api-sports.io/",
        "live_param": "live=all",
        "date_param": "date=",
        "team_home_key": ["teams", "home", "name"],
        "team_away_key": ["teams", "away", "name"],
        "score_home_key": ["goals", "home"],
        "score_away_key": ["goals", "away"],
        "status_key": ["fixture", "status", "short"],
        "elapsed_key": ["fixture", "status", "elapsed"],
        "league_key": ["league", "name"]
    },
    "Tennis": {
        "base_url": "https://v3.tennis.api-sports.io/",
        "live_param": "live=all",          # À confirmer → souvent "games?live=all" ou "fixtures/live"
        "date_param": "date=",
        "team_home_key": ["players", "home", "name"],  # ou "player1"
        "team_away_key": ["players", "away", "name"],
        "score_home_key": ["scores", "home"],
        "score_away_key": ["scores", "away"],
        "status_key": ["status"],
        "elapsed_key": None,
        "league_key": ["tournament", "name"]
    },
    "NBA": {
        "base_url": "https://v3.basketball.api-sports.io/",
        "live_param": "live=all",
        "date_param": "date=",
        "team_home_key": ["teams", "home", "name"],
        "team_away_key": ["teams", "away", "name"],
        "score_home_key": ["scores", "home"],
        "score_away_key": ["scores", "away"],
        "status_key": ["status", "short"],
        "elapsed_key": ["status", "clock"],
        "league_key": ["league", "name"]
    }
}

# ─── Fonction pour récupérer les données ────────────────────────────────────
@st.cache_data(ttl=30)  # Cache 30s pour éviter de spammer l'API
def get_matches(sport):
    config = SPORT_CONFIG[sport]
    today = date.today().strftime("%Y-%m-%d")
    headers = {"x-apisports-key": API_KEY}
    
    matches = []
    
    # 1. Live
    try:
        url_live = f"{config['base_url']}fixtures?{config['live_param']}"
        resp = requests.get(url_live, headers=headers, timeout=12)
        if resp.status_code == 200:
            matches.extend(resp.json().get("response", []))
    except Exception as e:
        st.error(f"Live {sport} erreur: {e}")
    
    # 2. Aujourd'hui
    try:
        url_date = f"{config['base_url']}fixtures?{config['date_param']}{today}"
        resp = requests.get(url_date, headers=headers, timeout=12)
        if resp.status_code == 200:
            today_matches = resp.json().get("response", [])
            # Évite doublons (live déjà inclus)
            live_ids = {m["fixture"]["id"] for m in matches if "fixture" in m}
            for m in today_matches:
                if "fixture" in m and m["fixture"]["id"] not in live_ids:
                    matches.append(m)
    except Exception as e:
        st.error(f"Date {sport} erreur: {e}")
    
    # Parsing adapté
    rows = []
    for match in matches:
        try:
            # Extraction par clés nested
            def get_nested(d, keys, default="?"):
                val = d
                for k in keys:
                    if isinstance(val, dict):
                        val = val.get(k, default)
                    else:
                        return default
                return val
            
            home = get_nested(match, config["team_home_key"])
            away = get_nested(match, config["team_away_key"])
            score_h = get_nested(match, config["score_home_key"])
            score_a = get_nested(match, config["score_away_key"])
            status = get_nested(match, config["status_key"])
            elapsed = get_nested(match, config["elapsed_key"]) or ""
            league = get_nested(match, config["league_key"])
            
            rows.append({
                "Match": f"{home} vs {away}",
                "Score": f"{score_h} - {score_a}" if score_h != "?" else "—",
                "Status": status,
                "Temps": elapsed,
                "Ligue": league
            })
        except:
            continue
    
    return pd.DataFrame(rows)

# ─── INTERFACE ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Sports + Pronos IA", layout="wide")
st.title("Scores en Direct – Football • Tennis • NBA ⚽🎾🏀")
st.caption("Données via API-Sports (refresh auto)")

sport = st.sidebar.selectbox("Sport", ["Football", "Tennis", "NBA"])
refresh_sec = st.sidebar.slider("Rafraîchissement (secondes)", 30, 180, 60, help="Ne descends pas trop bas pour respecter le quota gratuit ~100 req/jour")

placeholder = st.empty()

# Boucle live
while True:
    with placeholder.container():
        st.subheader(f"{sport} – Live & Matchs du jour")
        
        df = get_matches(sport)
        
        if df.empty:
            st.info("Pas de match trouvé aujourd'hui / en live, ou erreur API (clé, quota, endpoint ?)")
        else:
            # Highlight live
            def highlight(row):
                if row["Temps"] or "LIVE" in str(row["Status"]).upper():
                    return ['background-color: #d4edda'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                df.style.apply(highlight, axis=1),
                use_container_width=True,
                hide_index=True
            )
    
    time.sleep(refresh_sec)
    st.rerun()
