# Code Python efficace et fonctionnel pour analyser et pronostiquer des matchs NBA
# Utilise nba_api pour données réelles, LightGBM pour modélisation, et Pandas pour analyse
# Intégré dans une application Streamlit
# Étapes :
# 1. Fetch matchs du jour avec stats avancées
# 2. Analyse des données (pace, net rating, forme récente approximation)
# 3. Chargement ou entraînement d'un modèle simple LightGBM
# 4. Pronostic (proba victoire domicile)
# 5. Value bets simulés
# Installation requise (exécutez dans votre terminal) :
# pip install nba_api lightgbm pandas joblib streamlit

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, boxscoreadvancedv2
from nba_api.stats.static import teams
from lightgbm import LGBMClassifier
import joblib
import random
from datetime import date, timedelta
import numpy as np
import os
import streamlit as st

# Dictionnaire des équipes
team_list = teams.get_teams()
team_dict = {team['id']: team['abbreviation'] for team in team_list}

# ────────────────────────────────────────────────────────────────────────────
# 1. Fetch données réelles NBA (matchs du jour + stats avancées)
# ────────────────────────────────────────────────────────────────────────────
def fetch_nba_games_and_stats(days_back=0):
    # Date du jour ou jours précédents pour plus de données
    target_date = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
   
    try:
        # Scoreboard
        sb = scoreboardv2.ScoreboardV2(game_date=target_date)
        games_df = sb.get_data_frames()[0]
       
        if games_df.empty:
            return pd.DataFrame()
       
        rows = []
        for _, row in games_df.iterrows():
            game_id = row.get('GAME_ID', '')
           
            # Boxscore avancé
            stats = {}
            try:
                box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
                dfs = box.get_data_frames()
                if len(dfs) > 1 and not dfs[1].empty:
                    team_stats = dfs[1] # Team stats
                    home_id = row.get('HOME_TEAM_ID')
                    away_id = row.get('VISITOR_TEAM_ID')
                   
                    home_row = team_stats[team_stats['TEAM_ID'] == home_id]
                    away_row = team_stats[team_stats['TEAM_ID'] == away_id]
                   
                    if not home_row.empty:
                        stats['home_pace'] = home_row['PACE'].values[0]
                        stats['home_net'] = home_row['NET_RATING'].values[0]
                        stats['home_efg'] = home_row['EFG_PCT'].values[0]
                    if not away_row.empty:
                        stats['away_pace'] = away_row['PACE'].values[0]
                        stats['away_net'] = away_row['NET_RATING'].values[0]
                        stats['away_efg'] = away_row['EFG_PCT'].values[0]
            except:
                stats = {'error': 'Pas de stats avancées'}
           
            rows.append({
                'game_id': game_id,
                'home_id': row.get('HOME_TEAM_ID'),
                'away_id': row.get('VISITOR_TEAM_ID'),
                'home_score': row.get('PTS_HOME', '?'),
                'away_score': row.get('PTS_AWAY', '?'),
                'status': row.get('GAME_STATUS_TEXT', 'À venir'),
                'date': target_date,
                'stats': stats
            })
       
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        print(f"Erreur nba_api : {str(e)}")
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────
# 2. Analyse des données (features pour le modèle)
# ────────────────────────────────────────────────────────────────────────────
def analyze_data(df):
    df['pace_diff'] = df.apply(lambda r: r['stats'].get('home_pace', 0) - r['stats'].get('away_pace', 0), axis=1)
    df['net_diff'] = df.apply(lambda r: r['stats'].get('home_net', 0) - r['stats'].get('away_net', 0), axis=1)
    df['efg_diff'] = df.apply(lambda r: r['stats'].get('home_efg', 0) - r['stats'].get('away_efg', 0), axis=1)
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int) if 'home_score' in df else np.nan # Target pour entraînement
   
    return df

# ────────────────────────────────────────────────────────────────────────────
# 3. Modèle LightGBM (chargement ou entraînement)
# ────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "nba_model.pkl"
def get_model(df_train):
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Modèle chargé depuis fichier")
            return model
        except:
            print("Échec chargement → entraînement nouveau modèle")
    # Entraînement simple si pas de fichier (sur données disponibles)
    if df_train.empty or df_train['home_win'].isna().all():
        print("Pas de données pour entraînement → modèle dummy")
        model = LGBMClassifier(n_estimators=50, random_state=42)
        X_dummy = pd.DataFrame(np.random.rand(100, 3), columns=['pace_diff', 'net_diff', 'efg_diff'])
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
    else:
        features = ['pace_diff', 'net_diff', 'efg_diff']
        X = df_train[features].fillna(0)
        y = df_train['home_win']
        model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
        model.fit(X, y)
   
    joblib.dump(model, MODEL_PATH)
    print("Nouveau modèle entraîné et sauvegardé")
    return model

# ────────────────────────────────────────────────────────────────────────────
# 4. Pronostic & Value Bets
# ────────────────────────────────────────────────────────────────────────────
def make_predictions(df, model):
    features = ['pace_diff', 'net_diff', 'efg_diff']
    X = df[features].fillna(0)
    df['proba_home'] = model.predict_proba(X)[:, 1] # Proba victoire domicile
    return df

def calculate_value_bets(df, value_threshold=0.05):
    df['cote_home_sim'] = df['proba_home'].apply(lambda p: round(1 / p * random.uniform(0.92, 0.98), 2) if p > 0.1 else 10.0)
    df['value'] = df['proba_home'] * df['cote_home_sim'] - 1
    df['value_pct'] = df['value'] * 100
    df['is_value_bet'] = df['value'] > value_threshold
    return df

# ────────────────────────────────────────────────────────────────────────────
# 5. Exécution principale dans Streamlit
# ────────────────────────────────────────────────────────────────────────────
st.title("Pronostiqueur de Matchs NBA")

# Fetch données (du jour + 7 jours précédents pour plus d'analyse)
with st.spinner("Récupération des données NBA..."):
    all_data = pd.DataFrame()
    for d in range(0, 8):
        day_data = fetch_nba_games_and_stats(d)
        all_data = pd.concat([all_data, day_data], ignore_index=True)
   
    if all_data.empty:
        st.warning("Pas de données réelles – utilisation de données simulées")
        all_data = pd.DataFrame([
            {'game_id': 'sim1', 'home_id': 1610612747, 'away_id': 1610612738, 'home_score': 120, 'away_score': 115, 'status': 'Final', 'stats': {'home_pace': 98.5, 'home_net': 5.2, 'away_pace': 96.0, 'away_net': 2.1, 'home_efg': 0.55, 'away_efg': 0.52}},
            {'game_id': 'sim2', 'home_id': 1610612743, 'away_id': 1610612756, 'home_score': 105, 'away_score': 110, 'status': 'Final', 'stats': {'home_pace': 100.0, 'home_net': 3.8, 'away_pace': 99.5, 'away_net': 4.5, 'home_efg': 0.53, 'away_efg': 0.56}},
        ])
   
    # Ajout des noms d'équipes
    all_data['home_team'] = all_data['home_id'].map(team_dict)
    all_data['away_team'] = all_data['away_id'].map(team_dict)
    all_data['match'] = all_data['home_team'] + ' vs ' + all_data['away_team']
   
    analyzed_data = analyze_data(all_data)
    model = get_model(analyzed_data)
   
    predictions = make_predictions(analyzed_data, model)
    value_bets = calculate_value_bets(predictions)

# Affichage dans Streamlit
st.subheader("Données analysées et pronostics :")
display_df = value_bets[['match', 'proba_home', 'cote_home_sim', 'value_pct', 'is_value_bet']].copy()
display_df['proba_home'] = display_df['proba_home'].apply(lambda x: f"{x:.2%}")
display_df['value_pct'] = display_df['value_pct'].apply(lambda x: f"{x:.1f}%")
st.dataframe(display_df)

safest = value_bets.loc[value_bets['proba_home'].idxmax()]
st.success(f"Pronostic le plus sûr : Victoire domicile pour {safest['match']} à {safest['proba_home']:.2%} (value : {safest['value_pct']:.1f}%)")

st.info("Note : Pour les matchs futurs, les stats avancées ne sont pas disponibles, donc les prédictions sont basées sur des valeurs par défaut (0). Ce modèle est un exemple simplifié et non destiné à des paris réels.")
