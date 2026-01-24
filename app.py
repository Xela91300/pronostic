import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, boxscoreadvancedv2
import joblib
import random
import time
from datetime import datetime, date

# Mapping ID → Nom équipe (complet et stable)
TEAM_MAPPING = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612751: "Brooklyn Nets",
    1610612766: "Charlotte Hornets",
    1610612741: "Chicago Bulls",
    1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612754: "Indiana Pacers",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans",
    1610612752: "New York Knicks",
    1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612764: "Washington Wizards"
}

st.set_page_config(page_title="Pronos NBA + Boxscore Avancé", layout="wide")
st.title("Pronostics NBA – Boxscore Avancé & Value Bets")
st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')} CET")

refresh_sec = st.sidebar.slider("Rafraîchissement (s)", 120, 600, 300)
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 5, 20, 10)

placeholder = st.empty()

# Chargement modèle (optionnel)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("nba_model.pkl")
        st.success("Modèle LightGBM chargé !")
        return model
    except:
        st.warning("Modèle non trouvé → simulation proba")
        return None

model = load_model()

# ─── Récup matchs + boxscore avancé ────────────────────────────────────────
@st.cache_data(ttl=refresh_sec - 30)
def get_nba_data():
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"))
        games_df = sb.get_data_frames()[0]

        if games_df.empty:
            return pd.DataFrame()

        rows = []
        for _, row in games_df.iterrows():
            away_id = row.get('VISITOR_TEAM_ID')
            home_id = row.get('HOME_TEAM_ID')
            away_team = TEAM_MAPPING.get(away_id, "Équipe visiteuse inconnue")
            home_team = TEAM_MAPPING.get(home_id, "Équipe domicile inconnue")

            score = f"{row.get('HOME_TEAM_SCORE', '?')} - {row.get('VISITOR_TEAM_SCORE', '?')}" if row.get('HOME_TEAM_SCORE') is not None else "-"
            status = row.get('GAME_STATUS_TEXT', 'À venir')
            game_id = row.get('GAME_ID', '')

            # Boxscore avancé seulement si match en cours ou terminé
            advanced_stats = ""
            if game_id and "Final" in status or "Q" in status or "Half" in status:
                try:
                    box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
                    dfs = box.get_data_frames()
                    if len(dfs) > 0 and not dfs[0].empty:
                        player_stats = dfs[0]  # PlayerAdvancedStats souvent en position 0
                        if 'PACE' in player_stats.columns:
                            pace = player_stats['PACE'].mean()
                            advanced_stats += f"Pace : {pace:.1f} "
                        if 'NET_RATING' in player_stats.columns:
                            net = player_stats['NET_RATING'].mean()
                            advanced_stats += f"Net Rating : {net:.1f}"
                        if not advanced_stats:
                            advanced_stats = "Stats avancées disponibles mais pas de Pace/NetRating"
                    else:
                        advanced_stats = "Pas de stats avancées récupérées"
                except Exception as box_err:
                    advanced_stats = f"Erreur boxscore : {str(box_err)}"

            rows.append({
                "match": f"{away_team} @ {home_team}",
                "score": score,
                "status": status,
                "box_advanced": advanced_stats if advanced_stats else "Pas de boxscore avancé (match à venir ou erreur)"
            })

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Erreur globale : {str(e)}")
        return pd.DataFrame()

# ─── Prediction & Value ────────────────────────────────────────────────────
def predict_home_win_proba(row):
    if model is None:
        return round(random.uniform(0.50, 0.80), 3)
    try:
        features = pd.DataFrame([{"home_adv": 1.0}])
        return round(model.predict_proba(features)[0][1], 3)
    except:
        return 0.60

def add_value_bets(df):
    df["proba_home"] = df.apply(predict_home_win_proba, axis=1)
    df["cote_home_sim"] = df["proba_home"].apply(lambda p: round(1 / (p * random.uniform(0.90, 0.97)), 2) if p > 0.1 else 5.0)
    df["value"] = df["proba_home"] * df["cote_home_sim"] - 1
    df["value_pct"] = df["value"] * 100
    return df

# ─── Boucle live ───────────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df_nba = get_nba_data()
        
        if df_nba.empty:
            st.info("Aucun match NBA aujourd'hui ou erreur → simulation")
            df_nba = pd.DataFrame([
                {"match": "Lakers @ Celtics (exemple)", "score": "112-108", "status": "Final", "box_advanced": "Pace : 98.5 | Net Rating : +5.2"},
                {"match": "Nuggets @ Suns (exemple)", "score": "-", "status": "À venir", "box_advanced": "Pas de boxscore"}
            ])
        
        df_nba = add_value_bets(df_nba)
        
        # Pronostic le plus sûr
        if not df_nba.empty:
            safest = df_nba.loc[df_nba["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : Victoire domicile **{safest['match']}** → {safest['proba_home']:.0%}")
            st.markdown(f"Score : {safest['score']} | Statut : {safest['status']}")
            if safest['box_advanced']:
                st.markdown(f"**Stats avancées** : {safest['box_advanced']}")
        
        # Tableau
        disp = df_nba[["match", "score", "status", "box_advanced", "proba_home", "cote_home_sim", "value_pct"]].copy()
        disp = disp.rename(columns={
            "match": "Match",
            "score": "Score",
            "status": "Statut",
            "box_advanced": "Boxscore Avancé",
            "proba_home": "Proba Domicile",
            "cote_home_sim": "Cote D simulée",
            "value_pct": "Value %"
        })
        disp["Proba Domicile"] = disp["Proba Domicile"].apply(lambda x: f"{x:.0%}")
        disp["Cote D simulée"] = disp["Cote D simulée"].round(2)
        disp["Value %"] = disp["Value %"].round(1).apply(lambda x: f"+{x}%" if x > value_threshold else f"{x}%")
        
        def highlight(row):
            try:
                val = float(row["Value %"][:-1])
                if val > value_threshold:
                    return ["background-color: #d4edda"] * len(row)
            except:
                pass
            return [""] * len(row)
        
        st.dataframe(disp.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)
        
        st.caption("Données via nba_api (scoreboard + boxscoreadvancedv2) – Stats avancées seulement si match en cours/terminé")
    
    time.sleep(refresh_sec)
    st.rerun()
