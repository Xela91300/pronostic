import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, boxscoreadvancedv2
import joblib
import random
import time
from datetime import datetime, date

# Mapping ID équipe → Nom complet (à jour)
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

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pronos IA NBA Réel + Boxscore Avancé", layout="wide")
st.title("Pronostics NBA – Boxscore Avancé via nba_api + Value Bets")
st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')} CET | Massy, FR")

refresh_sec = st.sidebar.slider("Rafraîchissement (secondes)", 120, 600, 300)
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 5, 20, 10)

placeholder = st.empty()

# ─── MODELE LIGHTGBM (optionnel) ───────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("nba_model.pkl")
        st.success("Modèle LightGBM chargé !")
        return model
    except:
        st.warning("Modèle non trouvé → simulation proba activée")
        return None

model = load_model()

# ─── RECUP MATCHS DU JOUR + BOXSCORE AVANCÉ ────────────────────────────────
@st.cache_data(ttl=refresh_sec - 30)
def get_nba_games_today():
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"))
        games_df = sb.get_data_frames()[0]  # GameHeader

        if games_df.empty:
            return pd.DataFrame()

        rows = []
        for _, row in games_df.iterrows():
            # Noms via mapping ID
            away_id = row.get('VISITOR_TEAM_ID')
            home_id = row.get('HOME_TEAM_ID')
            away_team = TEAM_MAPPING.get(away_id, f"Équipe visiteuse ID {away_id}")
            home_team = TEAM_MAPPING.get(home_id, f"Équipe domicile ID {home_id}")

            score = f"{row.get('HOME_TEAM_SCORE', '?')} - {row.get('VISITOR_TEAM_SCORE', '?')}" if row.get('HOME_TEAM_SCORE') is not None else "-"
            status = row.get('GAME_STATUS_TEXT', 'À venir')
            game_id = row.get('GAME_ID', '')

            # Boxscore avancé pour ce match (si disponible)
            advanced_stats = ""
            try:
                if game_id and status != 'À venir':
                    box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
                    player_stats = box.get_data_frames()[0]  # PlayerAdvancedStats
                    if not player_stats.empty:
                        # Exemple : top net rating ou pace
                        pace = player_stats['PACE'].mean() if 'PACE' in player_stats.columns else "?"
                        net_rating = player_stats['NET_RATING'].mean() if 'NET_RATING' in player_stats.columns else "?"
                        advanced_stats = f"Pace : {pace:.1f} | Net Rating : {net_rating:.1f}"
            except Exception as box_err:
                advanced_stats = f"Erreur boxscore : {str(box_err)}"

            rows.append({
                "match": f"{away_team} @ {home_team}",
                "home_team": home_team,
                "away_team": away_team,
                "score": score,
                "status": status,
                "game_id": game_id,
                "box_advanced": advanced_stats
            })

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        st.error(f"Erreur globale nba_api : {str(e)}")
        return pd.DataFrame()

# ─── PROBA PREDICTION ──────────────────────────────────────────────────────
def predict_home_win_proba(row):
    if model is None:
        return round(random.uniform(0.50, 0.80), 3)
    
    try:
        features = pd.DataFrame([{"home_adv": 1.0}])
        proba = model.predict_proba(features)[0][1]
        return round(proba, 3)
    except:
        return 0.60

# ─── VALUE BETS ────────────────────────────────────────────────────────────
def add_value_bets(df):
    df["proba_home"] = df.apply(predict_home_win_proba, axis=1)
    df["cote_home_sim"] = df["proba_home"].apply(lambda p: round(1 / (p * random.uniform(0.90, 0.97)), 2) if p > 0.1 else 5.0)
    df["value"] = df["proba_home"] * df["cote_home_sim"] - 1
    df["value_pct"] = df["value"] * 100
    return df

# ─── BOUCLE PRINCIPALE ─────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df_nba = get_nba_games_today()
        
        if df_nba.empty:
            st.info("Aucun match NBA trouvé aujourd'hui ou erreur nba_api → simulation activée")
            df_nba = pd.DataFrame([
                {"match": "Los Angeles Lakers @ Boston Celtics (exemple)", "score": "112-108", "status": "Final", "box_advanced": "Pace : 98.5 | Net Rating : +5.2"},
                {"match": "Denver Nuggets @ Phoenix Suns (exemple)", "score": "-", "status": "À venir", "box_advanced": ""}
            ])
        
        df_nba = add_value_bets(df_nba)
        
        # Pronostic le plus sûr
        if not df_nba.empty:
            safest = df_nba.loc[df_nba["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : Victoire domicile **{safest['match']}** → {safest['proba_home']:.0%}")
            st.markdown(f"Score actuel : {safest['score']} | Statut : {safest['status']}")
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
        
        st.caption("Données via nba_api (scoreboard + boxscore advanced) – Proba via LightGBM ou simulation – Value = (proba × cote) - 1")
    
    time.sleep(refresh_sec)
    st.rerun()
