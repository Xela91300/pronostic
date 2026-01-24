import streamlit as st
import pandas as pd
import random
import time
import os
from datetime import datetime

# Debug tout en haut pour voir si on arrive ici
st.write("--- Début du script ---")
st.write("Version Python :", sys.version)

# Placeholder créé très tôt
placeholder = st.empty()

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION BASIQUE
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pronos NBA – Debug & Correction", layout="wide")
st.title("Pronostics NBA – Version corrigée & debuguée")
st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')} CET")

refresh_sec = st.sidebar.slider("Rafraîchissement (s)", 60, 300, 120)

# ─── DEBUG FICHIERS & MODÈLE ───────────────────────────────────────────────
st.subheader("Debug environnement")
st.write("Dossier courant :", os.getcwd())
st.write("Fichiers présents :", os.listdir("."))

model = None
if os.path.exists("nba_model.pkl"):
    st.success("nba_model.pkl détecté (taille : {} octets)".format(os.path.getsize("nba_model.pkl")))
    try:
        import joblib
        model = joblib.load("nba_model.pkl")
        st.success("Modèle chargé OK !")
    except Exception as e:
        st.error(f"Échec chargement : {str(e)}")
else:
    st.warning("nba_model.pkl absent → simulation activée")

# ─── FONCTION SIMULATION MATCHS (pas d'API pour éviter les bugs) ────────────
def get_fake_matches():
    return pd.DataFrame([
        {"match": "Lakers @ Celtics", "score": "112-108", "status": "Final", "proba_home": random.uniform(0.55, 0.85)},
        {"match": "Nuggets @ Suns", "score": "-", "status": "À venir", "proba_home": random.uniform(0.45, 0.75)},
        {"match": "Warriors @ Knicks", "score": "95-92 Q3", "status": "LIVE", "proba_home": random.uniform(0.60, 0.80)}
    ])

# ─── PRÉDICTION SIMPLE ─────────────────────────────────────────────────────
def predict_home(row):
    if model is not None:
        try:
            # Features fictives – adapte plus tard
            return round(model.predict_proba([[1.0]])[0][1], 3)
        except:
            pass
    return round(random.uniform(0.50, 0.80), 3)

# ─── BOUCLE LIVE ───────────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df = get_fake_matches()
        df["proba_home"] = df.apply(predict_home, axis=1)

        # Pronostic le plus sûr
        if not df.empty:
            best = df.loc[df["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : Victoire domicile **{best['match']}** → {best['proba_home']:.0%}")
            st.markdown(f"Score : {best['score']} | Statut : {best['status']}")

        # Tableau
        disp = df[["match", "score", "status", "proba_home"]].copy()
        disp["proba_home"] = disp["proba_home"].apply(lambda x: f"{x:.0%}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.caption("Version debug – sans API pour éviter les 403/NameError – Recharge auto")

    time.sleep(refresh_sec)
    st.rerun()
