import streamlit as st
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, boxscoreadvancedv2
import joblib
import random
import time
import os
from datetime import datetime, date

# Mapping ID équipe → Nom complet
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

st.set_page_config(page_title="Pronos NBA Réel + Boxscore", layout="wide")
st.title("Pronostics NBA – Matchs réels & Boxscore Avancé")
st.caption(f"Mis à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')} CET")

refresh_sec = st.sidebar.slider("Rafraîchissement (secondes)", 120, 600, 300)
value_threshold = st.sidebar.slider("Seuil Value Bet (%)", 5, 20, 10)

placeholder = st.empty()

# Debug rapide
st.write("Fichiers présents dans le dossier :", os.listdir("."))

# ─── CHARGEMENT DU MODELE ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "nba_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Le fichier {model_path} n'est pas trouvé dans le dossier racine")
        return None
    try:
        model = joblib.load(model_path)
        st.success(f"Modèle LightGBM chargé avec succès ! (taille : {os.path.getsize(model_path)} octets)")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None

model = load_model()

# ─── RÉCUP MATCHS + BOXSCORE ───────────────────────────────────────────────
@st.cache_data(ttl=refresh_sec - 30)
def get_nba_games():
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date.today().strftime("%Y-%m-%d"))
        games_df = sb.get_data_frames()[0]

        if games_df.empty:
            st.info("Aucun match NBA aujourd'hui")
            return pd.DataFrame()

        rows = []
        for _, row in games_df.iterrows():
            away_id = row.get('VISITOR_TEAM_ID')
            home_id = row.get('HOME_TEAM_ID')
            away = TEAM_MAPPING.get(away_id, f"Visiteur ID {away_id}")
            home = TEAM_MAPPING.get(home_id, f"Domicile ID {home_id}")

            score = f"{row.get('HOME_TEAM_SCORE', '?')} - {row.get('VISITOR_TEAM_SCORE', '?')}" \
                if row.get('HOME_TEAM_SCORE') is not None else "-"

            status = row.get('GAME_STATUS_TEXT', 'À venir')
            game_id = row.get('GAME_ID', '')

            box_info = "Pas de boxscore avancé"
            if game_id and ("Final" in status or "Q" in status or "Half" in status):
                try:
                    box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
                    dfs = box.get_data_frames()
                    if len(dfs) > 0 and not dfs[0].empty:
                        stats = dfs[0]
                        pace = stats['PACE'].mean() if 'PACE' in stats.columns else "?"
                        net = stats['NET_RATING'].mean() if 'NET_RATING' in stats.columns else "?"
                        box_info = f"Pace : {pace:.1f} | Net Rating : {net:.1f}"
                except Exception as box_err:
                    box_info = f"Erreur boxscore : {str(box_err)[:80]}..."

            rows.append({
                "match": f"{away} @ {home}",
                "score": score,
                "status": status,
                "box_advanced": box_info
            })

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Erreur globale nba_api : {str(e)}")
        return pd.DataFrame()

# ─── PREDICTION & VALUE BET ────────────────────────────────────────────────
def predict_home_proba(row):
    if model is None:
        return round(random.uniform(0.50, 0.80), 3)
    try:
        features = pd.DataFrame([{"home_adv": 1.0}])
        return round(model.predict_proba(features)[0][1], 3)
    except Exception as e:
        st.warning(f"Erreur prédiction : {str(e)} → simulation")
        return 0.60

def add_value_bets(df):
    df["proba_home"] = df.apply(predict_home_proba, axis=1)
    df["cote_home"] = df["proba_home"].apply(lambda p: round(1 / (p * random.uniform(0.90, 0.97)), 2) if p > 0.1 else 5.0)
    df["value_pct"] = (df["proba_home"] * df["cote_home"] - 1) * 100
    return df

# ─── BOUCLE PRINCIPALE ─────────────────────────────────────────────────────
while True:
    with placeholder.container():
        df = get_nba_games()

        if df.empty:
            st.info("Aucun match trouvé → simulation activée")
            df = pd.DataFrame([
                {"match": "Lakers @ Celtics (exemple)", "score": "112-108", "status": "Final", "box_advanced": "Pace : 98.5 | Net Rating : +5.2"},
                {"match": "Nuggets @ Suns (exemple)", "score": "-", "status": "À venir", "box_advanced": "Pas de données"}
            ])

        df = add_value_bets(df)

        # Pronostic le plus sûr
        if not df.empty:
            best = df.loc[df["proba_home"].idxmax()]
            st.success(f"**Pronostic le plus sûr** : {best['match']} (domicile) → {best['proba_home']:.0%}")
            st.markdown(f"Score : {best['score']} | Statut : {best['status']}")
            if best['box_advanced']:
                st.markdown(f"**Boxscore avancé** : {best['box_advanced']}")

        # Tableau
        disp = df[["match", "score", "status", "box_advanced", "proba_home", "cote_home", "value_pct"]].copy()
        disp = disp.rename(columns={
            "match": "Match",
            "score": "Score",
            "status": "Statut",
            "box_advanced": "Box Avancé",
            "proba_home": "Proba Domicile",
            "cote_home": "Cote D",
            "value_pct": "Value %"
        })
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

        st.caption("Données via nba_api – Box avancé seulement si match en cours/terminé – Proba via LightGBM ou simulation")

    time.sleep(refresh_sec)
    st.rerun()
