import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv2
import joblib
import random
import time
from datetime import datetime, date

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pronos IA NBA Réel (nba_api)", layout="wide")
st.title("Pronostics NBA – Données réelles via nba_api + Value Bets")
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
        st.warning("Modèle non trouvé → simulation proba")
        return None

model = load_model()

# ─── RECUP MATCHS DU JOUR via nba_api ──────────────────────────────────────
@st.cache_data(ttl=refresh_sec - 30)
def get_nba_games_today():
    try:
        # Scoreboard du jour
        sb = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"))
        games = sb.get_data_frames()[0]  # GameHeader
        
        if games.empty:
            return pd.DataFrame()
        
        # Ajoute status et score live si disponible
        rows = []
        for _, row in games.iterrows():
            status = row.get('GAME_STATUS_TEXT', 'À venir')
            score = f"{row.get('HOME_TEAM_SCORE', '?')} - {row.get('VISITOR_TEAM_SCORE', '?')}" if row.get('HOME_TEAM_SCORE') else "-"
            
            rows.append({
                "match": f"{row['VISITOR_TEAM_NAME']} @ {row['HOME_TEAM_NAME']}",
                "home_team": row['HOME_TEAM_NAME'],
                "away_team": row['VISITOR_TEAM_NAME'],
                "score": score,
                "status": status,
                "game_id": row['GAME_ID']
            })
        
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Erreur nba_api : {str(e)}")
        return pd.DataFrame()

# ─── PROBA PREDICTION ──────────────────────────────────────────────────────
def predict_home_win_proba(row):
    if model is None:
        # Simulation : home advantage + random
        return round(random.uniform(0.50, 0.80), 3)
    
    try:
        # Features dummy – adapte à ton entraînement réel
        features = pd.DataFrame([{"home_adv": 1.0, "dummy_stat": random.uniform(0, 10)}])
        proba = model.predict_proba(features)[0][1]  # home win classe 1
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

# ─── BOUCLE LIVE ───────────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df_nba = get_nba_games_today()
        
        if df_nba.empty:
            st.info("Aucun match NBA aujourd'hui ou erreur nba_api → simulation activée")
            df_nba = pd.DataFrame([
                {"match": "Exemple : Lakers @ Celtics", "score": "112-108", "status": "Final"},
                {"match": "Nuggets @ Suns", "score": "-", "status": "À venir"}
            ])
        
        df_nba = add_value_bets(df_nba)
        
        # Pronostic le plus sûr
        if not df_nba.empty:
            safest = df_nba.loc[df_nba["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : Victoire domicile **{safest['match']}** → {safest['proba_home']:.0%}")
            st.markdown(f"Score : {safest['score']} | Statut : {safest['status']}")
        
        # Tableau
        disp = df_nba[["match", "score", "status", "proba_home", "cote_home_sim", "value_pct"]].copy()
        disp.columns = ["Match", "Score", "Statut", "Proba Domicile", "Cote D simulée", "Value %"]
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
        
        st.caption("Données via nba_api (officiel NBA) – Proba via LightGBM ou simulation – Value = (proba × cote) - 1")
    
    time.sleep(refresh_sec)
    st.rerun()
