import streamlit as st
import pandas as pd
import joblib
import random
import time
import os
from datetime import datetime

# Mapping ID → Nom équipe (stable)
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

st.set_page_config(page_title="Pronos NBA – Version Stable", layout="wide")
st.title("Pronostics NBA – Version corrigée et stable")
st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')} CET")

refresh_sec = st.sidebar.slider("Rafraîchissement (secondes)", 60, 300, 120)
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 5, 20, 10)

placeholder = st.empty()

# Debug simple
st.write("Fichiers présents :", os.listdir("."))

# ─── CHARGEMENT MODELE ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "nba_model.pkl"
    if not os.path.exists(model_path):
        st.warning(f"{model_path} non trouvé → simulation activée")
        return None
    try:
        model = joblib.load(model_path)
        st.success("Modèle chargé !")
        return model
    except Exception as e:
        st.error(f"Erreur chargement : {str(e)}")
        return None

model = load_model()

# ─── MATCHS SIMULÉS (pas d'API pour éviter les bugs actuels) ────────────────
def get_matches():
    return pd.DataFrame([
        {"match": "Los Angeles Lakers @ Boston Celtics", "score": "112-108", "status": "Final"},
        {"match": "Denver Nuggets @ Phoenix Suns", "score": "-", "status": "À venir"},
        {"match": "Golden State Warriors @ New York Knicks", "score": "95-92 Q3", "status": "LIVE"}
    ])

# ─── PRÉDICTION ────────────────────────────────────────────────────────────
def predict_home(row):
    if model is None:
        return round(random.uniform(0.50, 0.80), 3)
    try:
        features = pd.DataFrame([{"home_adv": 1.0}])
        return round(model.predict_proba(features)[0][1], 3)
    except:
        return 0.60

# ─── VALUE BETS ────────────────────────────────────────────────────────────
def add_value(df):
    df["proba_home"] = df.apply(predict_home, axis=1)
    df["cote_sim"] = df["proba_home"].apply(lambda p: round(1 / (p * random.uniform(0.90, 0.97)), 2) if p > 0.1 else 5.0)
    df["value_pct"] = (df["proba_home"] * df["cote_sim"] - 1) * 100
    return df

# ─── BOUCLE ────────────────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df = get_matches()
        df = add_value(df)

        if not df.empty:
            best = df.loc[df["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : {best['match']} (domicile) → {best['proba_home']:.0%}")
            st.markdown(f"Score : {best['score']} | Statut : {best['status']}")

        disp = df[["match", "score", "status", "proba_home", "cote_sim", "value_pct"]].copy()
        disp.columns = ["Match", "Score", "Statut", "Proba Domicile", "Cote D", "Value %"]
        disp["Proba Domicile"] = disp["Proba Domicile"].apply(lambda x: f"{x:.0%}")
        disp["Cote D"] = disp["Cote D"].round(2)
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

        st.caption("Version sans API pour stabilité – Ajoute nba_api quand l'import est OK")

    time.sleep(refresh_sec)
    st.rerun()
