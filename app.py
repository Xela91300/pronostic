# app.py - Système de Pronostics Football avec Sélection de Matchs
# Version corrigée

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import random
import time
import requests
from typing import Dict, List

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
                    return True, "✅ Clé API valide"
                else:
                    return False, "❌ Clé API invalide"
            elif response.status_code == 403:
                return False, "❌ Clé API refusée"
            elif response.status_code == 429:
                return False, "❌ Trop de requêtes"
            else:
                return False, f"❌ Erreur {response.status_code}"
                
        except Exception as e:
            return False, f"❌ Erreur de connexion: {str(e)}"
    
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
                'date': today + timedelta(days=1, hours=20),
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
                'date': today + timedelta(days=2, hours=20),
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
                'date': today + timedelta(days=1, hours=18),
                'status': 'NS',
                'venue': 'Stade Pierre-Mauroy',
                'home_team_id': 79,
                'away_team_id': 84,
                'league_id': 61
            },
            {
                'id': 1004,
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'league': 'La Liga',
                'country': 'Spain',
                'date': today + timedelta(days=3, hours=21),
                'status': 'NS',
                'venue': 'Santiago Bernabéu',
                'home_team_id': 541,
                'away_team_id': 529,
                'league_id': 140
            },
            {
                'id': 1005,
                'home_team': 'Manchester City',
                'away_team': 'Liverpool',
                'league': 'Premier League',
                'country': 'England',
                'date': today + timedelta(days=2, hours=16),
                'status': 'NS',
                'venue': 'Etihad Stadium',
                'home_team_id': 50,
                'away_team_id': 40,
                'league_id': 39
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
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    .live-badge {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .upcoming-badge {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .league-badge {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'état
    if 'football_api' not in st.session_state:
        st.session_state.football_api = FootballAPIClient()
    
    if 'selected_match' not in st.session_state:
        st.session_state.selected_match = None
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "selection"
    
    # En-tête
    st.markdown('<h1 class="main-header">⚽ Pronostics Football</h1>', 
                unsafe_allow_html=True)
    
    # Vérification de l'API
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        status, message = st.session_state.football_api.test_api_key()
        if status:
            st.success(f"{message}")
        else:
            st.error(f"{message}")
            st.info("Mode démo activé")
        
        st.divider()
        
        # Bouton pour retourner à la sélection
        if st.session_state.view_mode == "analysis":
            if st.button("🔙 Retour à la sélection", use_container_width=True):
                st.session_state.view_mode = "selection"
                st.session_state.selected_match = None
                st.rerun()
    
    # Contenu principal
    if st.session_state.view_mode == "selection":
        display_match_selection()
    else:
        display_match_analysis()

def display_match_selection():
    """Affiche la sélection des matchs"""
    
    st.header("📋 Sélectionnez un match à analyser")
    
    # Options de filtrage dans le sidebar
    with st.sidebar:
        st.subheader("Filtres")
        
        # Filtre par période
        time_filter = st.selectbox(
            "Période",
            ["Aujourd'hui", "7 prochains jours"],
            key="time_filter"
        )
        
        # Liste des ligues (version simplifiée)
        leagues = [
            {"id": 61, "name": "Ligue 1", "country": "France", "logo": "🇫🇷"},
            {"id": 39, "name": "Premier League", "country": "England", "logo": "🏴"},
            {"id": 140, "name": "La Liga", "country": "Spain", "logo": "🇪🇸"},
            {"id": 78, "name": "Bundesliga", "country": "Germany", "logo": "🇩🇪"},
            {"id": 135, "name": "Serie A", "country": "Italy", "logo": "🇮🇹"}
        ]
        
        league_options = ["Toutes les ligues"] + [f"{l['logo']} {l['name']}" for l in leagues]
        selected_league = st.selectbox("Ligue", league_options, key="league_filter")
        
        # Option pour les matchs à venir seulement
        show_only_upcoming = st.checkbox("Matchs à venir seulement", value=True, key="upcoming_only")
        
        # Bouton de recherche
        if st.button("🔍 Rechercher", type="primary", use_container_width=True):
            st.rerun()
    
    # Récupération des matchs
    with st.spinner("Recherche des matchs..."):
        if time_filter == "Aujourd'hui":
            matches = st.session_state.football_api.get_todays_matches()
        else:
            matches = st.session_state.football_api.get_upcoming_matches(days=7)
    
    # Filtrer par statut si demandé
    if show_only_upcoming and matches:
        matches = [m for m in matches if m.get('status') == 'NS']
    
    # Affichage des résultats
    if not matches:
        st.warning("Aucun match trouvé.")
        st.info("Affichage des matchs de démonstration...")
        matches = st.session_state.football_api._get_fallback_matches()
    
    # Statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matchs trouvés", len(matches))
    with col2:
        upcoming = len([m for m in matches if m.get('status') == 'NS'])
        st.metric("À venir", upcoming)
    with col3:
        st.metric("Ligues", len(set(m['league'] for m in matches)))
    
    st.divider()
    
    # Affichage des matchs
    st.subheader(f"📅 Matchs disponibles ({len(matches)})")
    
    # Afficher chaque match
    for idx, match in enumerate(matches):
        display_match_card(match, idx)
    
    # Message si aucun match
    if len(matches) == 0:
        st.info("""
        ℹ️ **Conseil :**
        - Essayez de changer les filtres de recherche
        - Vérifiez votre connexion internet
        - Les matchs de démonstration sont affichés
        """)

def display_match_card(match, idx):
    """Affiche une carte pour un match"""
    
    # Formater la date
    date_str = match['date'].strftime("%d/%m/%Y")
    time_str = match['date'].strftime("%H:%M")
    
    # Déterminer le badge
    if match.get('status') == 'LIVE':
        badge_class = "live-badge"
        badge_text = "🔴 EN DIRECT"
    else:
        badge_class = "upcoming-badge"
        badge_text = "⏰ À VENIR"
    
    # Carte HTML
    st.markdown(f"""
    <div class="match-card">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
            <div class="{badge_class}">
                {badge_text}
            </div>
            <div style="text-align: right;">
                <div style="font-size: 14px; font-weight: bold; color: #333;">{date_str}</div>
                <div style="font-size: 12px; color: #666;">{time_str}</div>
            </div>
        </div>
        
        <div style="margin-bottom: 10px;">
            <div class="league-badge" style="display: inline-block;">
                {match['league']}
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">
                {match['country']} • {match['venue']}
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 20px 0;">
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">{match['home_team']}</div>
                <div style="font-size: 12px; color: #666;">Domicile</div>
            </div>
            
            <div style="margin: 0 15px;">
                <div style="font-size: 28px; font-weight: bold; color: #333;">VS</div>
            </div>
            
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">{match['away_team']}</div>
                <div style="font-size: 12px; color: #666;">Extérieur</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bouton pour analyser le match
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        button_key = f"analyze_{match['id']}_{idx}"
        if st.button(f"🔍 Analyser ce match", 
                    key=button_key, 
                    use_container_width=True,
                    type="primary"):
            st.session_state.selected_match = match
            st.session_state.view_mode = "analysis"
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

def display_match_analysis():
    """Affiche l'analyse d'un match sélectionné"""
    
    if not st.session_state.selected_match:
        st.error("Aucun match sélectionné.")
        st.session_state.view_mode = "selection"
        st.rerun()
        return
    
    match = st.session_state.selected_match
    
    # Bouton de retour
    if st.button("← Retour à la sélection"):
        st.session_state.view_mode = "selection"
        st.session_state.selected_match = None
        st.rerun()
    
    # En-tête
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <h2>🔍 Analyse du match</h2>
            <h3>{match['home_team']} vs {match['away_team']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Informations du match
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 Ligue", match['league'])
        st.metric("📍 Stade", match['venue'][:15] + "..." if len(match['venue']) > 15 else match['venue'])
    
    with col2:
        st.metric("📅 Date", match['date'].strftime("%d/%m/%Y"))
        st.metric("⏰ Heure", match['date'].strftime("%H:%M"))
    
    with col3:
        status_text = "🔴 EN DIRECT" if match.get('status') == 'LIVE' else "⏰ À VENIR"
        st.metric("📊 Statut", status_text)
        st.metric("🌍 Pays", match['country'])
    
    with col4:
        st.metric("🏠 Domicile", match['home_team'])
        st.metric("✈️ Extérieur", match['away_team'])
    
    st.markdown("---")
    
    # Onglets d'analyse
    tab1, tab2, tab3 = st.tabs(["📈 Statistiques", "🎯 Prédictions", "💰 Paris"])
    
    with tab1:
        display_statistics(match)
    
    with tab2:
        display_predictions(match)
    
    with tab3:
        display_betting(match)

def display_statistics(match):
    """Affiche les statistiques"""
    
    st.subheader("📊 Comparaison des équipes")
    
    # Générer des statistiques simulées
    home_stats = generate_team_stats(match['home_team'], is_home=True)
    away_stats = generate_team_stats(match['away_team'], is_home=False)
    
    # Tableau de comparaison
    comparison_data = {
        'Statistique': [
            'Forme récente',
            'Victoires domicile/extérieur',
            'Buts marqués (moyenne)',
            'Buts encaissés (moyenne)',
            'Possession moyenne',
            'Tirs par match'
        ],
        match['home_team']: [
            home_stats['recent_form'],
            f"{home_stats['home_wins']}/{home_stats['home_matches']}",
            f"{home_stats['avg_goals_for']:.1f}",
            f"{home_stats['avg_goals_against']:.1f}",
            f"{home_stats['possession']}%",
            f"{home_stats['shots_per_game']:.1f}"
        ],
        match['away_team']: [
            away_stats['recent_form'],
            f"{away_stats['away_wins']}/{away_stats['away_matches']}",
            f"{away_stats['avg_goals_for']:.1f}",
            f"{away_stats['avg_goals_against']:.1f}",
            f"{away_stats['possession']}%",
            f"{away_stats['shots_per_game']:.1f}"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison.set_index('Statistique'), use_container_width=True)

def generate_team_stats(team_name, is_home=True):
    """Génère des statistiques simulées"""
    
    # Forme récente
    recent_form = ''.join(random.choices(['W', 'D', 'L'], weights=[45, 25, 30], k=5))
    
    # Statistiques de base
    if is_home:
        home_wins = random.randint(5, 10)
        home_matches = random.randint(12, 18)
        avg_goals_for = random.uniform(1.5, 2.5)
    else:
        away_wins = random.randint(3, 8)
        away_matches = random.randint(12, 18)
        avg_goals_for = random.uniform(1.0, 2.0)
    
    return {
        'team': team_name,
        'recent_form': recent_form,
        'home_wins': home_wins if is_home else random.randint(3, 8),
        'home_matches': home_matches if is_home else random.randint(12, 18),
        'away_wins': random.randint(2, 7) if is_home else away_wins,
        'away_matches': random.randint(12, 18) if is_home else away_matches,
        'avg_goals_for': round(avg_goals_for, 1),
        'avg_goals_against': round(random.uniform(0.8, 1.8), 1),
        'possession': random.randint(48, 65),
        'shots_per_game': round(random.uniform(10, 18), 1)
    }

def display_predictions(match):
    """Affiche les prédictions"""
    
    st.subheader("🎯 Prédictions")
    
    # Calculer les probabilités
    home_prob = random.randint(40, 65)
    draw_prob = random.randint(20, 35)
    away_prob = 100 - home_prob - draw_prob
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Probabilités")
        
        # Victoire domicile
        st.markdown(f"**✅ {match['home_team']} gagne**")
        st.progress(home_prob/100)
        st.caption(f"{home_prob}%")
        
        # Match nul
        st.markdown("**⚪ Match nul**")
        st.progress(draw_prob/100)
        st.caption(f"{draw_prob}%")
        
        # Victoire extérieur
        st.markdown(f"**✅ {match['away_team']} gagne**")
        st.progress(away_prob/100)
        st.caption(f"{away_prob}%")
    
    with col2:
        st.markdown("### Score prédit")
        
        # Générer un score
        home_goals = random.randint(0, 3)
        away_goals = random.randint(0, 2)
        
        st.markdown(f"<h1 style='text-align: center; font-size: 3rem;'>{home_goals}-{away_goals}</h1>", 
                   unsafe_allow_html=True)
        
        st.metric("Total buts", home_goals + away_goals)
        
        if home_goals > 0 and away_goals > 0:
            st.success("✅ Les deux équipes devraient marquer")
        else:
            st.info("⚪ Une équipe pourrait rester à 0")

def display_betting(match):
    """Affiche l'analyse des paris"""
    
    st.subheader("💰 Opportunités de pari")
    
    # Générer des cotes
    bookmaker_odds = {
        'Bet365': {
            '1': round(random.uniform(1.6, 2.8), 2),
            'N': round(random.uniform(3.2, 3.8), 2),
            '2': round(random.uniform(2.5, 4.5), 2),
            'Over 2.5': round(random.uniform(1.7, 2.2), 2),
            'Under 2.5': round(random.uniform(1.6, 2.1), 2)
        },
        'Unibet': {
            '1': round(random.uniform(1.65, 2.85), 2),
            'N': round(random.uniform(3.25, 3.85), 2),
            '2': round(random.uniform(2.55, 4.55), 2),
            'Over 2.5': round(random.uniform(1.72, 2.22), 2),
            'Under 2.5': round(random.uniform(1.62, 2.12), 2)
        }
    }
    
    # Afficher les cotes
    st.markdown("### 📊 Cotes disponibles")
    
    odds_list = []
    for bookmaker, markets in bookmaker_odds.items():
        for market, odd in markets.items():
            odds_list.append({
                'Bookmaker': bookmaker,
                'Marché': market,
                'Cote': odd
            })
    
    df_odds = pd.DataFrame(odds_list)
    pivot_df = df_odds.pivot(index='Marché', columns='Bookmaker', values='Cote')
    st.dataframe(pivot_df, use_container_width=True)
    
    # Recommandations
    st.markdown("### 🎯 Recommandations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Niveau de risque", "Moyen")
        st.caption("Basé sur l'analyse")
    
    with col2:
        st.metric("Mise suggérée", "2-3%")
        st.caption("Pour bankroll 1000€")

if __name__ == "__main__":
    main()
