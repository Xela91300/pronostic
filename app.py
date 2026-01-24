# app.py – Version 2025-01-24 – nba_api stable + debug

import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
import joblib
import random
import time
import os
from datetime import datetime, date

# ─── Constantes & Mapping équipes ───────────────────────────────────────────
TEAM_MAPPING = {
    1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics", 1610612751: "Brooklyn Nets",
    1610612766: "Charlotte Hornets", 1610612741: "Chicago Bulls", 1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets", 1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors", 1610612745: "Houston Rockets", 1610612754: "Indiana Pacers",
    1610612746: "LA Clippers", 1610612747: "Los Angeles Lakers", 1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat", 1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks", 1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic", 1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings", 1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors", 1610612762: "Utah Jazz", 1610612764: "Washington Wizards"
}

MODEL_PATH = "nba_model.pkl"

# ─── Chargement modèle ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Modèle {MODEL_PATH} non trouvé → simulation activée")
        return None
    try:
        m = joblib.load(MODEL_PATH)
        st.success("Modèle LightGBM chargé")
        return m
    except Exception as e:
        st.error(f"Échec chargement modèle : {str(e)[:120]}")
        return None

model = load_model()

# ─── Récup matchs du jour (scoreboardv2) – Version ultra-stable ────────────
@st.cache_data(ttl=119)
def fetch_nba_games():
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"))
        df = sb.get_data_frames()[0]  # GameHeader

        if df.empty:
            return pd.DataFrame()

        # Colonnes les plus stables et utiles
        useful_cols = [
            'GAME_ID', 'GAME_STATUS_TEXT', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
            'HOME_TEAM_SCORE', 'VISITOR_TEAM_SCORE', 'GAME_STATUS_ID'
        ]
        df = df[useful_cols] if all(c in df.columns for c in useful_cols) else df

        # Mapping noms
        df['home_team'] = df['HOME_TEAM_ID'].map(TEAM_MAPPING).fillna("Domicile inconnu")
        df['away_team'] = df['VISITOR_TEAM_ID'].map(TEAM_MAPPING).fillna("Visiteur inconnu")

        df['match'] = df['away_team'] + " @ " + df['home_team']
        df['score'] = df.apply(
            lambda r: f"{r['HOME_TEAM_SCORE']} - {r['VISITOR_TEAM_SCORE']}" 
            if pd.notna(r['HOME_TEAM_SCORE']) else "-", axis=1
        )
        df['status'] = df['GAME_STATUS_TEXT'].fillna("À venir")

        return df[['match', 'score', 'status', 'home_team', 'away_team', 'GAME_ID']]

    except Exception as e:
        st.error(f"Erreur scoreboardv2 : {str(e)[:150]}")
        return pd.DataFrame()

# ─── Prédiction simple (améliorable plus tard) ─────────────────────────────
def predict_home_proba(row):
    if model is None:
        p = random.uniform(0.50, 0.80)
        if "home" in row['match'].lower(): p += 0.05
        return round(p, 3)
    try:
        # Features minimales – à enrichir plus tard
        return round(model.predict_proba([[1.0]])[0][1], 3)
    except:
        return 0.60

# ─── Value bets simulés ─────────────────────────────────────────────────────
def add_value_bets(df):
    df["proba_home"] = df.apply(predict_home_proba, axis=1)
    df["cote_home"] = df["proba_home"].apply(lambda p: round(1 / (p * random.uniform(0.90, 0.97)), 2) if p > 0.1 else 5.0)
    df["value_pct"] = (df["proba_home"] * df["cote_home"] - 1) * 100
    return df

# ─── Interface principale ──────────────────────────────────────────────────
while True:
    with placeholder.container():
        df = fetch_nba_games()

        if df.empty:
            st.info("Aucun match aujourd'hui ou erreur API → simulation")
            df = pd.DataFrame([
                {"match": "Lakers @ Celtics", "score": "112-108", "status": "Final"},
                {"match": "Nuggets @ Suns", "score": "-", "status": "À venir"}
            ])

        df = add_value_bets(df)

        # Pronostic le plus sûr
        if not df.empty:
            best = df.loc[df["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : {best['match']} → domicile gagne à {best['proba_home']:.0%}")
            st.markdown(f"Score : {best['score']} | Statut : {best['status']}")

        # Tableau
        disp = df[["match", "score", "status", "proba_home", "cote_home", "value_pct"]].copy()
        disp.columns = ["Match", "Score", "Statut", "Proba Domicile", "Cote simulée", "Value %"]
        disp["Proba Domicile"] = disp["Proba Domicile"].apply(lambda x: f"{x:.0%}")
        disp["Cote simulée"] = disp["Cote simulée"].round(2)
        disp["Value %"] = disp["Value %"].round(1).apply(lambda x: f"+{x}%" if x > value_threshold else f"{x}%")

        def highlight(row):
            try:
                v = float(row["Value %"][:-1])
                if v > value_threshold:
                    return ["background-color: #d4edda"] * len(row)
            except:
                pass
            return [""] * len(row)

        st.dataframe(disp.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)

        st.caption("Version stable – sans boxscore pour éviter les bugs – Proba simulée ou LightGBM")

    time.sleep(refresh_sec)
    st.rerun()
