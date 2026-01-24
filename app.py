import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, boxscoreadvancedv2
import joblib
import random
import time
import os
from datetime import datetime, date
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTES & MAPPING
# ────────────────────────────────────────────────────────────────────────────

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
VALUE_THRESHOLD_DEFAULT = 8
REFRESH_MIN = 120

# ────────────────────────────────────────────────────────────────────────────
# 1. Chargement & validation du modèle
# ────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_lightgbm_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable : {MODEL_PATH} n'existe pas dans le répertoire racine")
        return None

    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"Modèle chargé avec succès ({os.path.getsize(MODEL_PATH)} octets)")
        return model
    except Exception as e:
        st.error(f"Échec chargement modèle : {str(e)}\n→ Causes possibles : fichier corrompu, pickle protocol incompatible, version joblib/lightgbm différente")
        return None

# ────────────────────────────────────────────────────────────────────────────
# 2. Récupération des données (matchs + stats avancées)
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_MIN - 30)
def fetch_nba_games():
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"))
        games = sb.get_data_frames()[0]

        if games.empty:
            return pd.DataFrame()

        rows = []
        for _, g in games.iterrows():
            away_id = g.get('VISITOR_TEAM_ID')
            home_id = g.get('HOME_TEAM_ID')

            away_name = TEAM_MAPPING.get(away_id, f"ID {away_id}")
            home_name = TEAM_MAPPING.get(home_id, f"ID {home_id}")

            score = f"{g.get('HOME_TEAM_SCORE', '?')} - {g.get('VISITOR_TEAM_SCORE', '?')}" \
                if g.get('HOME_TEAM_SCORE') is not None else "-"

            status = g.get('GAME_STATUS_TEXT', 'À venir')
            game_id = g.get('GAME_ID', '')

            # Boxscore avancé
            box_data = {}
            if game_id and ("Final" in status or "Q" in status or "Half" in status):
                try:
                    box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
                    dfs = box.get_data_frames()
                    if len(dfs) > 0 and not dfs[0].empty:
                        team_stats = dfs[1]  # TeamAdvancedStats souvent en position 1
                        if not team_stats.empty:
                            home_row = team_stats[team_stats['TEAM_ID'] == home_id]
                            away_row = team_stats[team_stats['TEAM_ID'] == away_id]

                            if not home_row.empty:
                                box_data['home_pace'] = home_row['PACE'].values[0]
                                box_data['home_net'] = home_row['NET_RATING'].values[0]
                            if not away_row.empty:
                                box_data['away_pace'] = away_row['PACE'].values[0]
                                box_data['away_net'] = away_row['NET_RATING'].values[0]
                except Exception as e:
                    box_data['error'] = str(e)[:60]

            rows.append({
                "match": f"{away_name} @ {home_name}",
                "home": home_name,
                "away": away_name,
                "score": score,
                "status": status,
                "game_id": game_id,
                "box_data": box_data
            })

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Erreur récupération NBA : {str(e)}")
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────
# 3. Prédiction (avec features réelles si disponibles)
# ────────────────────────────────────────────────────────────────────────────

def predict_home_win_proba(match_row, model):
    if model is None:
        # Simulation basique pondérée
        proba = random.uniform(0.50, 0.80)
        if 'home' in match_row['match'].lower(): proba += 0.05
        return round(proba, 3)

    try:
        # Features réelles tirées du boxscore (si disponibles)
        features_dict = {
            "home_adv": 1.0,
            "pace_diff": 0.0,
            "net_rating_diff": 0.0
        }

        if 'box_data' in match_row and isinstance(match_row['box_data'], dict):
            bd = match_row['box_data']
            if 'home_pace' in bd and 'away_pace' in bd:
                features_dict["pace_diff"] = bd['home_pace'] - bd['away_pace']
            if 'home_net' in bd and 'away_net' in bd:
                features_dict["net_rating_diff"] = bd['home_net'] - bd['away_net']

        X = pd.DataFrame([features_dict])
        proba = model.predict_proba(X)[0][1]  # classe 1 = home win
        return round(proba, 3)

    except Exception as e:
        st.warning(f"Erreur prédiction : {str(e)}")
        return 0.60

# ────────────────────────────────────────────────────────────────────────────
# 4. Value Bets
# ────────────────────────────────────────────────────────────────────────────

def calculate_value_bets(df):
    df["proba_home"] = df.apply(lambda row: predict_home_win_proba(row, model), axis=1)
    df["cote_home_sim"] = df["proba_home"].apply(lambda p: round(1 / (p * random.uniform(0.90, 0.97)), 2) if p > 0.1 else 5.0)
    df["value"] = df["proba_home"] * df["cote_home_sim"] - 1
    df["value_pct"] = df["value"] * 100
    return df

# ────────────────────────────────────────────────────────────────────────────
# 5. INTERFACE & BOUCLE LIVE
# ────────────────────────────────────────────────────────────────────────────

while True:
    with placeholder.container():
        df_games = fetch_nba_games()

        if df_games.empty:
            st.info("Aucun match NBA aujourd'hui ou erreur API → mode simulation activé")
            df_games = pd.DataFrame([
                {"match": "Lakers @ Celtics (exemple)", "score": "112-108", "status": "Final", "box_data": {"home_pace": 98.5, "home_net": 5.2}},
                {"match": "Nuggets @ Suns (exemple)", "score": "-", "status": "À venir", "box_data": {}}
            ])

        df = calculate_value_bets(df_games)

        # Pronostic le plus sûr
        if not df.empty:
            safest = df.loc[df["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : {safest['match']} (domicile) → {safest['proba_home']:.0%}")
            st.markdown(f"Score : {safest['score']} | Statut : {safest['status']}")
            if safest['box_data']:
                st.markdown(f"**Stats avancées** : Pace domicile {safest['box_data'].get('home_pace', '?'):.1f} | Net Rating {safest['box_data'].get('home_net', '?'):.1f}")

        # Tableau
        disp = df[["match", "score", "status", "proba_home", "cote_home", "value_pct"]].copy()
        disp = disp.rename(columns={
            "match": "Match",
            "score": "Score",
            "status": "Statut",
            "proba_home": "Proba Domicile",
            "cote_home": "Cote D simulée",
            "value_pct": "Value %"
        })
        disp["Proba Domicile"] = disp["Proba Domicile"].apply(lambda x: f"{x:.0%}")
        disp["Cote D simulée"] = disp["Cote D simulée"].round(2)
        disp["Value %"] = disp["Value %"].round(1).apply(lambda x: f"+{x}%" if x > value_threshold else f"{x}%")

        def highlight_value(row):
            try:
                val = float(row["Value %"][:-1])
                if val > value_threshold:
                    return ["background-color: #d4edda"] * len(row)
            except:
                pass
            return [""] * len(row)

        st.dataframe(disp.style.apply(highlight_value, axis=1), use_container_width=True, hide_index=True)

        st.caption("Données via nba_api – Box avancé seulement pour matchs en cours/terminés – Proba via LightGBM ou simulation")

    time.sleep(refresh_sec)
    st.rerun()
