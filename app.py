import streamlit as st
import pandas as pd
import joblib
import random
import time
import os
from datetime import datetime

# PLACEHOLDER CRÉÉ TOUT EN HAUT POUR ÉVITER LE NameError
placeholder = st.empty()

st.set_page_config(page_title="Pronos NBA - Minimal", layout="wide")
st.title("Pronostics NBA - Version minimale et stable")

refresh_sec = st.sidebar.slider("Rafraîchissement (s)", 60, 300, 120)
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 5, 20, 10)

# Debug
st.write("Fichiers dans le dossier :", os.listdir("."))

# Chargement modèle
model = None
if os.path.exists("nba_model.pkl"):
    try:
        model = joblib.load("nba_model.pkl")
        st.success(f"Modèle chargé (taille : {os.path.getsize('nba_model.pkl')} octets)")
    except Exception as e:
        st.error(f"Erreur modèle : {str(e)}")
else:
    st.warning("nba_model.pkl non trouvé → simulation")

# Matchs exemples
df = pd.DataFrame([
    {"match": "Lakers @ Celtics", "score": "112-108", "status": "Final"},
    {"match": "Nuggets @ Suns", "score": "-", "status": "À venir"},
    {"match": "Warriors @ Knicks", "score": "95-92 Q3", "status": "LIVE"}
])

# Prédiction
df["proba_home"] = df.apply(lambda r: round(random.uniform(0.50, 0.80), 3) if model is None else 0.65, axis=1)

# Value bets
df["cote_home"] = df["proba_home"].apply(lambda p: round(1 / (p * random.uniform(0.90, 0.97)), 2))
df["value_pct"] = (df["proba_home"] * df["cote_home"] - 1) * 100

# Pronostic le plus sûr
if not df.empty:
    best = df.loc[df["proba_home"].idxmax()]
    st.success(f"**Pronostic le plus sûr** : {best['match']} → {best['proba_home']:.0%}")

# Tableau
disp = df[["match", "score", "status", "proba_home", "cote_home", "value_pct"]].copy()
disp.columns = ["Match", "Score", "Statut", "Proba Domicile", "Cote D", "Value %"]
disp["Proba Domicile"] = disp["Proba Domicile"].apply(lambda x: f"{x:.0%}")
disp["Cote D"] = disp["Cote D"].round(2)
disp["Value %"] = disp["Value %"].round(1).apply(lambda x: f"+{x}%" if x > value_threshold else f"{x}%")

st.dataframe(disp, use_container_width=True, hide_index=True)

st.caption("Version minimale sans API – Ajoute nba_api seulement quand ça tourne parfaitement")

# Rafraîchissement
time.sleep(refresh_sec)
st.rerun()
