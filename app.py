import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
import joblib
import random
import time
from datetime import datetime, date

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pronos IA NBA Réel (nba_api corrigé)", layout="wide")
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
        st.warning("Modèle non trouvé → simulation proba activée")
        return None

model = load_model()

# ─── RECUP MATCHS DU JOUR via nba_api (colonnes corrigées) ─────────────────
@st.cache_data(ttl=refresh_sec - 30)
def get_nba_games_today():
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"))
        games_df = sb.get_data_frames()[0]  # GameHeader est généralement le premier DF

        if games_df.empty:
            return pd.DataFrame()

        # Colonnes réelles typiques (adapté à la doc nba_api) :
        # VISITOR_TEAM_CITY + VISITOR_TEAM_NICKNAME, HOME_TEAM_CITY + HOME_TEAM_NICKNAME
        # ou parfois VISITOR_TEAM_NAME / HOME_TEAM_NAME
        rows = []
        for _, row in games_df.iterrows():
            # Construction sécurisée des noms d'équipe
            away_team = f"{row.get('VISITOR_TEAM_CITY', '')} {row.get('VISITOR_TEAM_NICKNAME', row.get('VISITOR_TEAM_NAME', '?'))}".strip()
            home_team = f"{row.get('HOME_TEAM_CITY', '')} {row.get('HOME_TEAM_NICKNAME', row.get('HOME_TEAM_NAME', '?'))}".strip()

            score = f"{row.get('HOME_TEAM_SCORE', '?')} - {row.get('VISITOR_TEAM_SCORE', '?')}" if row.get('HOME_TEAM_SCORE') else "-"
            status = row.get('GAME_STATUS_TEXT', 'À venir')

            rows.append({
                "match": f"{away_team} @ {home_team}",
                "home_team": home_team,
                "away_team": away_team,
                "score": score,
                "status": status,
                "game_id": row.get('GAME_ID', '')
            })

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        st.error(f"Erreur nba_api : {str(e)}")
        return pd.DataFrame()

# ─── PROBA PREDICTION ──────────────────────────────────────────────────────
def predict_home_win_proba(row):
    if model is None:
        # Simulation réaliste
        return round(random.uniform(0.50, 0.80), 3)
    
    try:
        # Features dummy – adapte à ton .pkl réel
        features = pd.DataFrame([{"home_adv": 1.0}])
        proba = model.predict_proba(features)[0][1]  # assume classe 1 = home win
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
        disp = disp.rename(columns={
            "match": "Match",
            "score": "Score",
            "status": "Statut",
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
        
        st.caption("Données via nba_api – Proba via LightGBM ou simulation – Value = (proba × cote) - 1")
    
    time.sleep(refresh_sec)
    st.rerun()
