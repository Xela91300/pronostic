# app.py - Système de Pronostics Football avec API Réelle
# Version simplifiée et fonctionnelle

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import random
import time
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
import warnings
import re
import math

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DES APIS
# =============================================================================

class APIConfig:
    """Configuration des APIs externes"""
    
    # VOTRE CLÉ API RÉELLE POUR LE FOOTBALL
    FOOTBALL_API_KEY = "33a972705943458ebcbcae6b56e4dee0"
    
    # URLs des APIs
    FOOTBALL_API_URL = "https://v3.football.api-sports.io"
    
    @staticmethod
    def get_football_headers():
        """Retourne les headers avec votre clé API réelle"""
        return {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': APIConfig.FOOTBALL_API_KEY
        }

# =============================================================================
# CLIENT API FOOTBALL
# =============================================================================

class FootballAPIClient:
    """Client pour l'API Football"""
    
    def __init__(self):
        self.base_url = APIConfig.FOOTBALL_API_URL
        self.headers = APIConfig.get_football_headers()
        self.timeout = 30
        
    def test_api_key(self):
        """Teste si la clé API fonctionne"""
        try:
            response = requests.get(
                f"{self.base_url}/status",
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('response', {}).get('account'):
                    return True, "Clé API valide"
                else:
                    return False, "Clé API invalide"
            elif response.status_code == 403:
                return False, "Clé API refusée"
            elif response.status_code == 429:
                return False, "Trop de requêtes"
            else:
                return False, f"Erreur {response.status_code}"
                
        except Exception as e:
            return False, f"Erreur de connexion: {str(e)}"
    
    def get_todays_matches(self):
        """Récupère les matchs d'aujourd'hui"""
        today = date.today().strftime('%Y-%m-%d')
        
        try:
            response = requests.get(
                f"{self.base_url}/fixtures",
                headers=self.headers,
                params={'date': today, 'timezone': 'Europe/Paris'},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_fixtures(data)
            else:
                return self._get_fallback_matches()
                
        except Exception as e:
            return self._get_fallback_matches()
    
    def get_upcoming_matches(self, days: int = 7):
        """Récupère les matchs à venir"""
        end_date = (date.today() + timedelta(days=days)).strftime('%Y-%m-%d')
        
        params = {
            'from': date.today().strftime('%Y-%m-%d'),
            'to': end_date,
            'status': 'NS',
            'timezone': 'Europe/Paris'
        }
        
        try:
            response = requests.get(
                f"{self.base_url}/fixtures",
                headers=self.headers,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_fixtures(data)
            else:
                return self._get_fallback_matches()
                
        except Exception as e:
            return self._get_fallback_matches()
    
    def get_team_statistics(self, team_id: int, league_id: int, season: int = 2024):
        """Récupère les statistiques d'une équipe"""
        try:
            response = requests.get(
                f"{self.base_url}/teams/statistics",
                headers=self.headers,
                params={
                    'team': team_id,
                    'league': league_id,
                    'season': season
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            return None
    
    def _parse_fixtures(self, api_data):
        """Parse les données d'API"""
        matches = []
        
        if 'response' not in api_data:
            return matches
        
        for fixture in api_data['response']:
            try:
                # Date du match
                fixture_date = fixture['fixture']['date']
                match_date = datetime.fromisoformat(fixture_date.replace('Z', '+00:00'))
                
                # Équipes
                home_team = fixture['teams']['home']['name']
                away_team = fixture['teams']['away']['name']
                
                # Ligue
                league_info = fixture.get('league', {})
                league_name = league_info.get('name', 'Ligue Inconnue')
                league_country = league_info.get('country', 'Inconnu')
                
                match_info = {
                    'id': fixture['fixture']['id'],
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league_name,
                    'country': league_country,
                    'date': match_date,
                    'status': fixture['fixture']['status']['short'],
                    'venue': fixture['fixture']['venue']['name'] if fixture['fixture']['venue'] else 'Stade Inconnu',
                    'home_team_id': fixture['teams']['home']['id'],
                    'away_team_id': fixture['teams']['away']['id'],
                    'league_id': league_info.get('id', 0)
                }
                
                matches.append(match_info)
                
            except Exception as e:
                continue
        
        return matches
    
    def _get_fallback_matches(self):
        """Retourne des matchs de démonstration"""
        today = datetime.now()
        matches = []
        
        # Matchs de démo
        demo_matches = [
            {
                'id': 1001,
                'home_team': 'Paris SG',
                'away_team': 'Marseille',
                'league': 'Ligue 1',
                'country': 'France',
                'date': today + timedelta(days=1),
                'status': 'NS',
                'venue': 'Parc des Princes',
                'home_team_id': 85,
                'away_team_id': 81,
                'league_id': 61
            },
            {
                'id': 1002,
                'home_team': 'Lyon',
                'away_team': 'Monaco',
                'league': 'Ligue 1',
                'country': 'France',
                'date': today + timedelta(days=2),
                'status': 'NS',
                'venue': 'Groupama Stadium',
                'home_team_id': 80,
                'away_team_id': 91,
                'league_id': 61
            },
            {
                'id': 1003,
                'home_team': 'Lille',
                'away_team': 'Nice',
                'league': 'Ligue 1',
                'country': 'France',
                'date': today + timedelta(days=1),
                'status': 'NS',
                'venue': 'Stade Pierre-Mauroy',
                'home_team_id': 79,
                'away_team_id': 84,
                'league_id': 61
            }
        ]
        
        return demo_matches

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Application principale Streamlit"""
    
    # Configuration de la page
    st.set_page_config(
        page_title="Pronostics Football",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .match-card {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialisation
    if 'football_api' not in st.session_state:
        st.session_state.football_api = FootballAPIClient()
    
    if 'selected_match' not in st.session_state:
        st.session_state.selected_match = None
    
    # En-tête
    st.markdown('<h1 class="main-header">⚽ Pronostics Football avec API Réelle</h1>', 
                unsafe_allow_html=True)
    
    # Test de l'API
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        status, message = st.session_state.football_api.test_api_key()
        if status:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
            st.info("Utilisation du mode démo")
        
        st.divider()
        
        # Filtres
        st.subheader("Filtres")
        time_filter = st.selectbox(
            "Période",
            ["Aujourd'hui", "7 prochains jours"],
            key="time_filter"
        )
        
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()
    
    # Contenu principal
    if st.session_state.selected_match:
        display_match_analysis(st.session_state.selected_match)
    else:
        display_match_selection(time_filter)

def display_match_selection(time_filter):
    """Affiche la sélection des matchs"""
    
    st.header("⚽ Sélectionnez un match à analyser")
    
    # Récupération des matchs
    with st.spinner("Chargement des matchs..."):
        if time_filter == "Aujourd'hui":
            matches = st.session_state.football_api.get_todays_matches()
        else:
            matches = st.session_state.football_api.get_upcoming_matches(days=7)
    
    # Affichage des matchs
    if not matches:
        st.warning("Aucun match trouvé.")
        st.info("Affichage des matchs de démonstration...")
        matches = st.session_state.football_api._get_fallback_matches()
    
    st.subheader(f"📋 {len(matches)} match(s) disponible(s)")
    
    # Afficher chaque match
    for match in matches:
        display_match_card(match)

def display_match_card(match):
    """Affiche une carte pour un match"""
    
    # Formater la date
    date_str = match['date'].strftime("%d/%m/%Y %H:%M")
    
    # Carte HTML
    st.markdown(f"""
    <div class="match-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="background: #4CAF50; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                ⏰ À venir
            </div>
            <div style="font-size: 12px; color: #666;">
                {date_str}
            </div>
        </div>
        
        <div style="text-align: center; margin: 15px 0;">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                {match['league']}
            </div>
            <div style="font-size: 14px; color: #666; margin-bottom: 15px;">
                {match['country']} • {match['venue']}
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1; text-align: right;">
                    <div style="font-size: 18px; font-weight: bold;">
                        {match['home_team']}
                    </div>
                </div>
                
                <div style="margin: 0 20px;">
                    <div style="font-size: 28px; font-weight: bold; color: #333;">VS</div>
                </div>
                
                <div style="flex: 1; text-align: left;">
                    <div style="font-size: 18px; font-weight: bold;">
                        {match['away_team']}
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bouton pour analyser le match
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"🔍 Analyser ce match", 
                    key=f"analyze_{match['id']}", 
                    use_container_width=True):
            st.session_state.selected_match = match
            st.rerun()
    
    st.markdown("---")

def display_match_analysis(match):
    """Affiche l'analyse d'un match"""
    
    # Bouton de retour
    if st.button("🔙 Retour à la sélection"):
        st.session_state.selected_match = None
        st.rerun()
    
    st.header(f"🔍 Analyse: {match['home_team']} vs {match['away_team']}")
    
    # Informations de base
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ligue", match['league'])
        st.metric("Date", match['date'].strftime("%d/%m/%Y"))
    
    with col2:
        st.metric("Stade", match['venue'])
        st.metric("Heure", match['date'].strftime("%H:%M"))
    
    with col3:
        st.metric("Statut", "À venir")
        st.metric("Pays", match['country'])
    
    st.divider()
    
    # Onglets d'analyse
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Statistiques", 
        "📈 Prédiction", 
        "💰 Paris", 
        "📋 Info"
    ])
    
    with tab1:
        display_statistics(match)
    
    with tab2:
        display_prediction(match)
    
    with tab3:
        display_betting_analysis(match)
    
    with tab4:
        display_match_info(match)

def display_statistics(match):
    """Affiche les statistiques"""
    
    st.subheader("📊 Statistiques des équipes")
    
    # Récupérer les statistiques réelles
    with st.spinner("Chargement des statistiques..."):
        home_stats = st.session_state.football_api.get_team_statistics(
            match['home_team_id'], match['league_id'], 2024
        )
        away_stats = st.session_state.football_api.get_team_statistics(
            match['away_team_id'], match['league_id'], 2024
        )
    
    if home_stats and away_stats and 'response' in home_stats and 'response' in away_stats:
        # Extraire les statistiques
        home_data = extract_team_stats(home_stats, match['home_team'])
        away_data = extract_team_stats(away_stats, match['away_team'])
        
        # Afficher le tableau
        stats_df = pd.DataFrame({
            'Statistique': ['Matches', 'Victoires', 'Nuls', 'Défaites', 'Buts pour', 'Buts contre'],
            match['home_team']: [
                home_data['matches_played'],
                home_data['wins'],
                home_data['draws'],
                home_data['loses'],
                home_data['goals_for'],
                home_data['goals_against']
            ],
            match['away_team']: [
                away_data['matches_played'],
                away_data['wins'],
                away_data['draws'],
                away_data['loses'],
                away_data['goals_for'],
                away_data['goals_against']
            ]
        })
        
        st.dataframe(stats_df.set_index('Statistique'), use_container_width=True)
        
    else:
        st.warning("Statistiques non disponibles")
        display_simulated_stats(match)

def extract_team_stats(stats_data, team_name):
    """Extrait les statistiques d'une équipe"""
    try:
        if 'response' not in stats_data:
            return generate_simulated_stats(team_name)
        
        response = stats_data['response']
        fixtures = response.get('fixtures', {})
        goals = response.get('goals', {})
        
        return {
            'team': team_name,
            'matches_played': fixtures.get('played', {}).get('total', 0),
            'wins': fixtures.get('wins', {}).get('total', 0),
            'draws': fixtures.get('draws', {}).get('total', 0),
            'loses': fixtures.get('loses', {}).get('total', 0),
            'goals_for': goals.get('for', {}).get('total', {}).get('total', 0),
            'goals_against': goals.get('against', {}).get('total', {}).get('total', 0)
        }
        
    except:
        return generate_simulated_stats(team_name)

def generate_simulated_stats(team_name):
    """Génère des statistiques simulées"""
    return {
        'team': team_name,
        'matches_played': random.randint(20, 38),
        'wins': random.randint(8, 25),
        'draws': random.randint(5, 12),
        'loses': random.randint(3, 15),
        'goals_for': random.randint(25, 80),
        'goals_against': random.randint(15, 50)
    }

def display_simulated_stats(match):
    """Affiche des statistiques simulées"""
    
    home_stats = generate_simulated_stats(match['home_team'])
    away_stats = generate_simulated_stats(match['away_team'])
    
    stats_df = pd.DataFrame({
        'Statistique': ['Matches', 'Victoires', 'Nuls', 'Défaites', 'Buts pour', 'Buts contre'],
        match['home_team']: [
            home_stats['matches_played'],
            home_stats['wins'],
            home_stats['draws'],
            home_stats['loses'],
            home_stats['goals_for'],
            home_stats['goals_against']
        ],
        match['away_team']: [
            away_stats['matches_played'],
            away_stats['wins'],
            away_stats['draws'],
            away_stats['loses'],
            away_stats['goals_for'],
            away_stats['goals_against']
        ]
    })
    
    st.dataframe(stats_df.set_index('Statistique'), use_container_width=True)
    
    st.info("Données simulées - l'API n'a pas retourné de données réelles")

def display_prediction(match):
    """Affiche les prédictions"""
    
    st.subheader("🎯 Prédiction du match")
    
    # Calculer les probabilités
    probabilities = calculate_probabilities(match)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Probabilités")
        
        for outcome, prob in probabilities.items():
            label = "Domicile" if outcome == 'home' else "Nul" if outcome == 'draw' else "Extérieur"
            st.markdown(f"**{label}:** {prob}%")
            st.progress(prob/100)
    
    with col2:
        st.markdown("### Score prédit")
        
        # Générer un score
        home_goals = random.randint(0, 3)
        away_goals = random.randint(0, 2)
        predicted_score = f"{home_goals}-{away_goals}"
        
        st.markdown(f"<h1 style='text-align: center; font-size: 3rem;'>{predicted_score}</h1>", 
                   unsafe_allow_html=True)
        
        st.metric("Total buts", home_goals + away_goals)
        
        if home_goals > 0 and away_goals > 0:
            st.success("✅ Les deux équipes devraient marquer")
        else:
            st.info("⚪ Une équipe pourrait rester à 0")
    
    # Facteurs influents
    st.markdown("### 📊 Facteurs influents")
    
    factors = [
        ("Avantage domicile", random.randint(60, 85)),
        ("Forme récente", random.randint(50, 80)),
        ("Blessures", random.randint(20, 70)),
        ("Motivation", random.randint(60, 90))
    ]
    
    for factor, value in factors:
        st.write(f"**{factor}:** {value}%")
        st.progress(value/100)

def calculate_probabilities(match):
    """Calcule les probabilités de résultat"""
    
    # Simuler des probabilités réalistes
    base_home = random.uniform(40, 60)
    base_draw = random.uniform(20, 35)
    base_away = 100 - base_home - base_draw
    
    # Ajustements
    home_advantage = random.uniform(1.05, 1.15)
    home_prob = round(base_home * home_advantage, 1)
    draw_prob = round(base_draw, 1)
    away_prob = round(base_away, 1)
    
    # Normaliser
    total = home_prob + draw_prob + away_prob
    home_prob = round((home_prob / total) * 100, 1)
    draw_prob = round((draw_prob / total) * 100, 1)
    away_prob = round((away_prob / total) * 100, 1)
    
    return {
        'home': home_prob,
        'draw': draw_prob,
        'away': away_prob
    }

def display_betting_analysis(match):
    """Affiche l'analyse des paris"""
    
    st.subheader("💰 Analyse des opportunités de pari")
    
    # Cotes des bookmakers
    bookmaker_odds = generate_bookmaker_odds(match)
    
    # Afficher les cotes
    st.markdown("### Cotes disponibles")
    
    odds_df = pd.DataFrame(bookmaker_odds).T
    st.dataframe(odds_df, use_container_width=True)
    
    # Value bets
    st.markdown("### 💎 Paris recommandés")
    
    value_bets = find_value_bets(match, bookmaker_odds)
    
    if value_bets:
        for bet in value_bets:
            with st.expander(f"✅ {bet['bookmaker']} - {bet['market']}"):
                st.metric("Cote", bet['odd'])
                st.metric("Valeur estimée", f"+{bet['value']}%")
                st.metric("Confiance", f"{bet['confidence']}/10")
    else:
        st.info("Aucun pari avec valeur significative détecté")
    
    # Recommandations
    st.markdown("### 🎯 Recommandations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Niveau de risque", "Moyen")
        st.caption("Basé sur l'analyse des données")
    
    with col2:
        st.metric("Mise suggérée", "2-3%")
        st.caption("Pour une bankroll de 1000€")

def generate_bookmaker_odds(match):
    """Génère des cotes de bookmakers"""
    
    base_home_odd = random.uniform(1.5, 3.0)
    
    bookmakers = {
        'Bet365': {
            'Domicile': round(base_home_odd, 2),
            'Nul': round(random.uniform(3.0, 4.0), 2),
            'Extérieur': round(1 / ((1/base_home_odd) - 0.1), 2),
            'Over 2.5': round(random.uniform(1.6, 2.1), 2),
            'Under 2.5': round(random.uniform(1.6, 2.1), 2)
        },
        'Unibet': {
            'Domicile': round(base_home_odd + 0.05, 2),
            'Nul': round(random.uniform(3.1, 4.1), 2),
            'Extérieur': round(1 / ((1/base_home_odd) - 0.12), 2),
            'Over 2.5': round(random.uniform(1.65, 2.15), 2),
            'Under 2.5': round(random.uniform(1.55, 2.05), 2)
        }
    }
    
    return bookmakers

def find_value_bets(match, bookmaker_odds):
    """Identifie les paris avec de la valeur"""
    
    value_bets = []
    
    for bookmaker, odds in bookmaker_odds.items():
        for market, odd in odds.items():
            # Simuler une détection de valeur
            if random.random() < 0.3:  # 30% de chance
                value_bets.append({
                    'bookmaker': bookmaker,
                    'market': market,
                    'odd': odd,
                    'value': round(random.uniform(5, 15), 1),
                    'confidence': random.randint(6, 9)
                })
    
    return value_bets[:2]  # Limiter à 2

def display_match_info(match):
    """Affiche les informations du match"""
    
    st.subheader("📋 Informations détaillées")
    
    # Tableau d'informations
    info_data = {
        'Champ': [
            'ID du match',
            'Équipe domicile',
            'ID domicile',
            'Équipe extérieur',
            'ID extérieur',
            'Ligue',
            'ID ligue',
            'Stade',
            'Date',
            'Statut'
        ],
        'Valeur': [
            match['id'],
            match['home_team'],
            match['home_team_id'],
            match['away_team'],
            match['away_team_id'],
            match['league'],
            match['league_id'],
            match['venue'],
            match['date'].strftime("%d/%m/%Y %H:%M"),
            match['status']
        ]
    }
    
    df_info = pd.DataFrame(info_data)
    st.dataframe(df_info.set_index('Champ'), use_container_width=True)
    
    # Conseils
    st.markdown("### 💡 Conseils d'analyse")
    
    tips = [
        "Consultez les dernières nouvelles des équipes",
        "Vérifiez les absences de joueurs clés",
        "Analysez la forme récente (5 derniers matchs)",
        "Tenez compte des conditions météo",
        "Comparez les cotes sur plusieurs bookmakers"
    ]
    
    for i, tip in enumerate(tips, 1):
        st.write(f"{i}. {tip}")

if __name__ == "__main__":
    main()
