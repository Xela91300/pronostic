import streamlit as st
import pandas as pd
import requests
import time
import random
from datetime import datetime

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ────────────────────────────────────────────────────────────────────────────

ODDS_API_KEY = "b9250f9ec1510f4136bdaca0b1f4f5cf"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

st.set_page_config(page_title="Pronos IA + Cotes Réelles", layout="wide")

st.title("Pronostics IA Temps Réel + Value Bets ⚽🏀🎾")
st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Rafraîchissement auto")

# Sidebar
with st.sidebar:
    st.header("Paramètres")
    refresh_interval = st.slider("Intervalle de rafraîchissement (secondes)", 60, 300, 120)
    region = st.selectbox("Région des bookmakers", ["eu", "uk", "us", "au"], index=0)
    value_threshold = st.slider("Seuil minimum pour Value Bet (%)", 3, 20, 8)

placeholder = st.empty()

# ────────────────────────────────────────────────────────────────────────────
# FONCTIONS
# ────────────────────────────────────────────────────────────────────────────

def fetch_odds_from_api():
    """Récupère les cotes via The Odds API"""
    url = (
        f"{ODDS_BASE_URL}/sports/soccer/odds/"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions={region}"
        f"&markets=h2h"
        f"&oddsFormat=decimal"
    )

    try:
        response = requests.get(url, timeout=12)
        st.session_state.api_status = response.status_code
        st.session_state.api_message = response.text[:300] if response.status_code != 200 else ""

        if response.status_code == 200:
            data = response.json()
            if not data:
                return pd.DataFrame()

            rows = []
            for event in data:
                home_team = event.get("home_team", "?")
                away_team = event.get("away_team", "?")
                bookmakers = event.get("bookmakers", [])

                if bookmakers and bookmakers[0].get("markets"):
                    market = bookmakers[0]["markets"][0]
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    rows.append({
                        "match": f"{home_team} vs {away_team}",
                        "ligue": event.get("sport_key", "?").replace("soccer_", "").upper(),
                        "commence_time": event.get("commence_time", "?"),
                        "cote_home": outcomes.get(home_team, 0.0),
                        "cote_draw": outcomes.get("Draw", 0.0),
                        "cote_away": outcomes.get(away_team, 0.0),
                    })

            return pd.DataFrame(rows)
        else:
            return pd.DataFrame()

    except Exception as e:
        st.session_state.api_status = -1
        st.session_state.api_message = str(e)
        return pd.DataFrame()


def add_ia_predictions(df):
    """Ajoute des probabilités IA simulées (remplace par ton modèle LightGBM plus tard)"""
    if df.empty:
        return df

    df["proba_home_ia"] = [round(random.uniform(0.42, 0.82), 3) for _ in range(len(df))]
    df["value_home"] = df["proba_home_ia"] * df["cote_home"] - 1
    df["value_pct"] = df["value_home"] * 100
    return df


# ────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE – RAFRAÎCHISSEMENT AUTOMATIQUE
# ────────────────────────────────────────────────────────────────────────────

while True:
    with placeholder.container():
        df_odds = fetch_odds_from_api()

        if df_odds.empty:
            st.info("Aucune donnée récupérée pour le moment.")
            if "api_status" in st.session_state:
                status = st.session_state.api_status
                if status == 401:
                    st.error("Erreur 401 → Clé API invalide ou non reconnue")
                elif status == 403:
                    st.error("Erreur 403 → Accès interdit (clé non activée ?)")
                elif status == 429:
                    st.error("Erreur 429 → Quota dépassé (500 req/mois sur plan gratuit)")
                elif status == -1:
                    st.error(f"Erreur réseau ou timeout : {st.session_state.api_message}")
                else:
                    st.error(f"Code HTTP inattendu : {status}")
                    if "api_message" in st.session_state:
                        st.code(st.session_state.api_message)
        else:
            df = add_ia_predictions(df_odds)

            st.subheader(f"Matchs trouvés : {len(df)} – Région : {region.upper()}")

            display_df = df[[
                "match", "ligue", "cote_home", "proba_home_ia", "value_pct"
            ]].copy()

            display_df["cote_home"] = display_df["cote_home"].round(2).astype(str)
            display_df["proba_home_ia"] = (display_df["proba_home_ia"] * 100).round(0).astype(int).astype(str) + " %"
            display_df["value_pct"] = display_df["value_pct"].round(1).apply(
                lambda x: f"+{x}%" if x > value_threshold else f"{x}%"
            )

            def style_value_bet(row):
                if row["value_pct"].startswith("+") and float(row["value_pct"][1:-1]) > value_threshold:
                    return ["background-color: #d4edda"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_df.style.apply(style_value_bet, axis=1),
                use_container_width=True,
                hide_index=True
            )

            st.caption("Value = (proba IA × cote domicile) - 1 → vert si opportunité détectée")

    time.sleep(refresh_interval)
    st.rerun()
