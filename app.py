import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
import time
from datetime import date
import random  # pour simulation si besoin

# ─── CONFIG ────────────────────────────────────────────────────────────────
API_KEY = "b9250f9ec1510f4136bdaca0b1f4f5cf"
BASE_URL = "https://v3.football.api-sports.io/"

st.set_page_config(page_title="Pronos IA Live + Value", layout="wide")
st.title("Pronostics IA Temps Réel + Value Bets ⚽")
st.caption(f"Mis à jour : {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')} | Refresh auto")

# Charger modèle ML (football)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("football_model.pkl")
        st.success("Modèle LightGBM chargé !")
        return model
    except:
        st.warning("Modèle non trouvé → simulation activée")
        return None

model = load_model()

# Sidebar
refresh_sec = st.sidebar.slider("Rafraîchissement (s)", 30, 180, 60)
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 3, 20, 5)

placeholder = st.empty()

# ─── RÉCUP MATCHS LIVE + JOUR via API ──────────────────────────────────────
@st.cache_data(ttl=refresh_sec - 5)
def fetch_matches():
    today = date.today().strftime("%Y-%m-%d")
    headers = {"x-apisports-key": API_KEY}
    
    matches = []
    
    # Live
    try:
        resp_live = requests.get(f"{BASE_URL}fixtures/live", headers=headers, timeout=10)
        if resp_live.status_code == 200:
            matches.extend(resp_live.json().get("response", []))
    except Exception as e:
        st.error(f"Erreur live: {e}")
    
    # Aujourd'hui
    try:
        resp_today = requests.get(f"{BASE_URL}fixtures?date={today}", headers=headers, timeout=10)
        if resp_today.status_code == 200:
            today_list = resp_today.json().get("response", [])
            live_ids = {m["fixture"]["id"] for m in matches}
            for m in today_list:
                if m["fixture"]["id"] not in live_ids:
                    matches.append(m)
    except Exception as e:
        st.error(f"Erreur today: {e}")
    
    if not matches:
        return pd.DataFrame()
    
    # Parsing simplifié
    rows = []
    for m in matches:
        fixture = m["fixture"]
        teams = m["teams"]
        goals = m["goals"]
        league = m["league"]
        
        rows.append({
            "match": f"{teams['home']['name']} vs {teams['away']['name']}",
            "ligue": league["name"],
            "status": fixture["status"]["short"],
            "elapsed": fixture["status"].get("elapsed", ""),
            "score_home": goals["home"] if goals["home"] is not None else 0,
            "score_away": goals["away"] if goals["away"] is not None else 0,
            # Features pour modèle (simplifiées – adapte à ton entraînement)
            "home_xG": random.uniform(0.5, 2.5),  # ← remplace par vraies si dispo via autre source
            "away_xG": random.uniform(0.3, 2.0),
            "form_home": random.randint(3, 10),
            "form_away": random.randint(2, 9)
        })
    
    df = pd.DataFrame(rows)
    return df

# ─── PREDICTION ML ─────────────────────────────────────────────────────────
def predict_proba(df):
    if model is None:
        df["proba_home"] = [random.uniform(0.40, 0.85) for _ in df.index]
    else:
        try:
            X = df[["home_xG", "away_xG", "form_home", "form_away"]].fillna(0)
            df["proba_home"] = model.predict_proba(X)[:, 0]  # assume classe 0 = home win
        except:
            df["proba_home"] = 0.50  # fallback si features mismatch
    return df

# ─── COTES SIMULÉES + VALUE ────────────────────────────────────────────────
def add_odds_and_value(df):
    df["cote_home_sim"] = df["proba_home"].apply(
        lambda p: round(1 / (p * random.uniform(0.92, 0.97)), 2) if p > 0.1 else 10.0
    )
    df["value"] = (df["proba_home"] * df["cote_home_sim"]) - 1
    df["value_pct"] = df["value"] * 100
    return df

# ─── BOUCLE LIVE ───────────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df = fetch_matches()
        
        if df.empty:
            st.info("Aucun match aujourd'hui ou erreur API (clé/quota/endpoint ?)")
        else:
            # Filtre ligue
            all_ligues = sorted(df["ligue"].unique())
            selected_ligues = st.multiselect(
                "Filtrer ligues",
                options=all_ligues,
                default=all_ligues[:5] if len(all_ligues) > 5 else all_ligues
            )
            if selected_ligues:
                df = df[df["ligue"].isin(selected_ligues)]
            
            if df.empty:
                st.info("Aucun match dans les ligues sélectionnées")
            else:
                df = predict_proba(df)
                df = add_odds_and_value(df)
                
                st.subheader(f"Matchs ({len(df)}) – {pd.Timestamp.now().strftime('%H:%M:%S')}")
                
                display_cols = ["match", "ligue", "status", "elapsed", "score_home", "score_away", "proba_home", "cote_home_sim", "value_pct"]
                disp = df[display_cols].copy()
                disp["proba_home"] = disp["proba_home"].apply(lambda x: f"{x:.0%}")
                disp["cote_home_sim"] = disp["cote_home_sim"].apply(lambda x: f"{x:.2f}")
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
                
                st.caption("Value = (proba × cote) - 1 → vert si > seuil | Cotes simulées (remplace par API odds si plan payant)")
    
    time.sleep(refresh_sec)
    st.rerun()
