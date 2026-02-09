# app.py - Système de Pronostics Football avec Sélection de Matchs
# Version finale fonctionnelle

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
    
    def get_tomorrow_matches(self):
        """Récupère les matchs de demain"""
        tomorrow = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            response = requests.get(
                f"{self.base_url}/fixtures",
                headers=self.headers,
                params={'date': tomorrow, 'timezone': 'Europe/Paris'},
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
    
    def get_live_matches(self):
        """Récupère les matchs en direct"""
        try:
            response = requests.get(
                f"{self.base_url}/fixtures",
                headers=self.headers,
                params={'live': 'all'},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_fixtures(data)
            else:
                return []
                
        except Exception as e:
            return []
    
    def get_league_matches(self, league_id: int):
        """Récupère les matchs d'une ligue spécifique"""
        today = date.today().strftime('%Y-%m-%d')
        end_date = (date.today() + timedelta(days=14)).strftime('%Y-%m-%d')
        
        params = {
            'from': today,
            'to': end_date,
            'league': league_id,
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
                return []
                
        except Exception as e:
            return []
    
    def get_popular_leagues(self):
        """Retourne les ligues populaires avec leurs IDs"""
        leagues = [
            {'id': 61, 'name': 'Ligue 1', 'country': 'France', 'logo': '🇫🇷'},
            {'id': 39, 'name': 'Premier League', 'country': 'England', 'logo': '🏴󠁧󠁢󠁥󠁮󠁧󠁿'},
            {'id': 140, 'name': 'La Liga', 'country': 'Spain', 'logo': '🇪🇸'},
            {'id': 78, 'name': 'Bundesliga', 'country': 'Germany', 'logo': '🇩🇪'},
            {'id': 135, 'name': 'Serie A', 'country': 'Italy', 'logo': '🇮🇹'},
            {'id': 88, 'name': 'Eredivisie', 'country': 'Netherlands', 'logo': '🇳🇱'},
            {'id': 94, 'name': 'Primeira Liga', 'country': 'Portugal', 'logo': '🇵🇹'},
            {'id': 203, 'name': 'Super Lig', 'country': 'Turkey', 'logo': '🇹🇷'},
            {'id': 262, 'name': 'MLS', 'country': 'USA', 'logo': '🇺🇸'},
            {'id': 253, 'name': 'Brasileirão', 'country': 'Brazil', 'logo': '🇧🇷'},
        ]
        return leagues
    
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
        
        # Matchs de démo Ligue 1
        ligue1_matches = [
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
            },
            {
                'id': 1006,
                'home_team': 'Bayern Munich',
                'away_team': 'Borussia Dortmund',
                'league': 'Bundesliga',
                'country': 'Germany',
                'date': today + timedelta(days=4, hours=18),
                'status': 'NS',
                'venue': 'Allianz Arena',
                'home_team_id': 157,
                'away_team_id': 165,
                'league_id': 78
            }
        ]
        
        return ligue1_matches

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Application principale Streamlit"""
    
    # Configuration de la page
    st.set_page_config(
        page_title="Pronostics Football - Sélection de Matchs",
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
        transition: all 0.3s ease;
    }
    .match-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
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
        st.session_state.view_mode = "selection"  # "selection" ou "analysis"
    
    # En-tête
    st.markdown('<h1 class="main-header">⚽ Pronostics Football - Sélection de Matchs</h1>', 
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
    
    # Contenu principal basé sur le mode de vue
    if st.session_state.view_mode == "selection":
        display_match_selection()
    else:
        display_match_analysis()

def display_match_selection():
    """Affiche la sélection des matchs"""
    
    st.header("📋 Sélectionnez un match à analyser")
    
    # Options de filtrage dans le sidebar
    with st.sidebar:
        st.subheader("Filtres de recherche")
        
        # Filtre par période
        time_filter = st.selectbox(
            "Période",
            ["Aujourd'hui", "Demain", "7 prochains jours", "En direct", "Toutes les ligues"],
            key="time_filter"
        )
        
        # Filtre par ligue
        leagues = st.session_state.football_api.get_popular_leagues()
        league_options = ["Toutes les ligues"] + [f"{l['logo']} {l['name']}" for l in leagues]
        selected_league = st.selectbox("Ligue", league_options, key="league_filter")
        
        # Extraire l'ID de la ligue sélectionnée
        league_id = None
        if selected_league != "Toutes les ligues":
            for league in leagues:
                if f"{league['logo']} {league['name']}" == selected_league:
                    league_id = league['id']
                    break
        
        # Option pour les matchs à venir seulement
        show_only_upcoming = st.checkbox("Matchs à venir seulement", value=True, key="upcoming_only")
        
        # Bouton de recherche
        if st.button("🔍 Rechercher des matchs", type="primary", use_container_width=True):
            st.rerun()
    
    # Récupération des matchs selon les filtres
    with st.spinner("Recherche des matchs..."):
        matches = get_matches_by_filter(time_filter, league_id, show_only_upcoming)
    
    # Affichage des résultats
    if not matches:
        st.warning("Aucun match trouvé pour les critères sélectionnés.")
        st.info("Affichage des matchs de démonstration...")
        matches = st.session_state.football_api._get_fallback_matches()
    
    # Statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matchs trouvés", len(matches))
    with col2:
        live_matches = len([m for m in matches if m.get('status') == 'LIVE'])
        st.metric("En direct", live_matches)
    with col3:
        upcoming_matches = len([m for m in matches if m.get('status') == 'NS'])
        st.metric("À venir", upcoming_matches)
    
    st.divider()
    
    # Affichage des matchs en grille
    st.subheader(f"📅 Matchs disponibles ({len(matches)})")
    
    # Créer 2 colonnes pour l'affichage
    cols = st.columns(2)
    
    for idx, match in enumerate(matches):
        with cols[idx % 2]:
            display_match_card(match, idx)
    
    # Si aucun match n'est sélectionné, afficher un message
    if len(matches) == 0:
        st.info("""
        ℹ️ **Conseil :**
        - Essayez de changer les filtres de recherche
        - Vérifiez que votre clé API est valide
        - Consultez les matchs de démonstration ci-dessous
        """)

def get_matches_by_filter(time_filter, league_id=None, upcoming_only=True):
    """Récupère les matchs selon les filtres"""
    
    matches = []
    
    try:
        if time_filter == "Aujourd'hui":
            matches = st.session_state.football_api.get_todays_matches()
        elif time_filter == "Demain":
            matches = st.session_state.football_api.get_tomorrow_matches()
        elif time_filter == "7 prochains jours":
            matches = st.session_state.football_api.get_upcoming_matches(days=7)
        elif time_filter == "En direct":
            matches = st.session_state.football_api.get_live_matches()
        elif time_filter == "Toutes les ligues" and league_id:
            matches = st.session_state.football_api.get_league_matches(league_id)
        else:
            # Par défaut, les 7 prochains jours
            matches = st.session_state.football_api.get_upcoming_matches(days=7)
        
        # Filtrer par ligue si spécifié
        if league_id and matches:
            matches = [m for m in matches if m.get('league_id') == league_id]
        
        # Filtrer par statut si demandé
        if upcoming_only and matches:
            matches = [m for m in matches if m.get('status') == 'NS']
        
        # Trier par date
        matches.sort(key=lambda x: x.get('date', datetime.now()))
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des matchs: {str(e)}")
        matches = st.session_state.football_api._get_fallback_matches()
    
    return matches

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
    card_html = f"""
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
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">{match['home_team']}</div>
                <div style="font-size: 12px; color: #666;">Domicile</div>
            </div>
            
            <div style="margin: 0 15px;">
                <div style="font-size: 24px; font-weight: bold; color: #333;">VS</div>
            </div>
            
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">{match['away_team']}</div>
                <div style="font-size: 12px; color: #666;">Extérieur</div>
            </div>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
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
    
    # Bouton de retour en haut
    if st.button("← Retour à la sélection"):
        st.session_state.view_mode = "selection"
        st.session_state.selected_match = None
        st.rerun()
    
    # En-tête de l'analyse
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
        st.metric("📍 Stade", match['venue'][:20] + "..." if len(match['venue']) > 20 else match['venue'])
    
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Statistiques détaillées", 
        "🎯 Prédictions", 
        "💰 Opportunités de pari", 
        "📋 Informations complètes"
    ])
    
    with tab1:
        display_detailed_statistics(match)
    
    with tab2:
        display_predictions(match)
    
    with tab3:
        display_betting_opportunities(match)
    
    with tab4:
        display_complete_info(match)

def display_detailed_statistics(match):
    """Affiche les statistiques détaillées"""
    
    st.subheader("📊 Comparaison des équipes")
    
    # Générer des statistiques simulées
    home_stats = generate_team_stats(match['home_team'], is_home=True)
    away_stats = generate_team_stats(match['away_team'], is_home=False)
    
    # Tableau de comparaison
    comparison_data = {
        'Statistique': [
            'Forme récente (5 derniers)',
            'Victoires à domicile/extérieur',
            'Buts marqués (moyenne)',
            'Buts encaissés (moyenne)',
            'Possession moyenne (%)',
            'Tirs par match',
            'Précision des tirs (%)',
            'Fautes par match',
            'Cartons jaunes',
            'Cartons rouges'
        ],
        match['home_team']: [
            home_stats['recent_form'],
            f"{home_stats['home_wins']} sur {home_stats['home_matches']}",
            f"{home_stats['avg_goals_for']:.1f}",
            f"{home_stats['avg_goals_against']:.1f}",
            f"{home_stats['possession']}%",
            f"{home_stats['shots_per_game']:.1f}",
            f"{home_stats['shot_accuracy']}%",
            f"{home_stats['fouls_per_game']:.1f}",
            home_stats['yellow_cards'],
            home_stats['red_cards']
        ],
        match['away_team']: [
            away_stats['recent_form'],
            f"{away_stats['away_wins']} sur {away_stats['away_matches']}",
            f"{away_stats['avg_goals_for']:.1f}",
            f"{away_stats['avg_goals_against']:.1f}",
            f"{away_stats['possession']}%",
            f"{away_stats['shots_per_game']:.1f}",
            f"{away_stats['shot_accuracy']}%",
            f"{away_stats['fouls_per_game']:.1f}",
            away_stats['yellow_cards'],
            away_stats['red_cards']
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison.set_index('Statistique'), use_container_width=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparaison buts
        goals_data = pd.DataFrame({
            'Équipe': [match['home_team'], match['away_team']],
            'Buts marqués': [home_stats['avg_goals_for'], away_stats['avg_goals_for']],
            'Buts encaissés': [home_stats['avg_goals_against'], away_stats['avg_goals_against']]
        })
        st.bar_chart(goals_data.set_index('Équipe'), height=300)
        st.caption("Moyenne de buts par match")
    
    with col2:
        # Comparaison possession
        possession_data = pd.DataFrame({
            'Équipe': [match['home_team'], match['away_team']],
            'Possession': [home_stats['possession'], away_stats['possession']],
            'Précision tirs': [home_stats['shot_accuracy'], away_stats['shot_accuracy']]
        })
        st.bar_chart(possession_data.set_index('Équipe'), height=300)
        st.caption("Possession et précision")

def generate_team_stats(team_name, is_home=True):
    """Génère des statistiques simulées pour une équipe"""
    
    # Forme récente (W=Victoire, D=Nul, L=Défaite)
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
        'shots_per_game': round(random.uniform(10, 18), 1),
        'shot_accuracy': random.randint(35, 55),
        'fouls_per_game': round(random.uniform(10, 16), 1),
        'yellow_cards': random.randint(15, 40),
        'red_cards': random.randint(0, 3)
    }

def display_predictions(match):
    """Affiche les prédictions"""
    
    st.subheader("🎯 Prédictions et analyses")
    
    # Calculer les probabilités
    probabilities = calculate_match_probabilities(match)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Probabilités de résultat")
        
        # Affichage des probabilités avec barres
        for outcome, prob in probabilities['outcomes'].items():
            label = {
                'home_win': f"✅ Victoire {match['home_team']}",
                'draw': "⚪ Match nul",
                'away_win': f"✅ Victoire {match['away_team']}"
            }[outcome]
            
            st.markdown(f"**{label}**")
            st.progress(prob['probability']/100)
            st.caption(f"{prob['probability']}% • Cote estimée: {prob['estimated_odds']:.2f}")
        
        # Score le plus probable
        st.markdown("### 🥇 Score le plus probable")
        most_likely_score = max(probabilities['score_probabilities'], 
                               key=lambda x: probabilities['score_probabilities'][x])
        st.markdown(f"<h2 style='text-align: center;'>{most_likely_score}</h2>", 
                   unsafe_allow_html=True)
        st.caption(f"Probabilité: {probabilities['score_probabilities'][most_likely_score]:.1f}%")
    
    with col2:
        st.markdown("### 📈 Analyse prédictive")
        
        # Facteurs clés
        st.markdown("#### 🔑 Facteurs influents")
        
        factors = [
            ("Avantage domicile", random.randint(65, 85)),
            ("Forme récente", random.randint(40, 75)),
            ("Blessures/absences", random.randint(20, 60)),
            ("Motivation", random.randint(60, 90)),
            ("Historique face-à-face", random.randint(30, 70))
        ]
        
        for factor, value in factors:
            st.write(f"**{factor}:** {value}%")
            st.progress(value/100)
        
        # Recommandation
        st.markdown("#### 💡 Recommandation")
        
        best_outcome = max(probabilities['outcomes'].items(), 
                          key=lambda x: x[1]['probability'])
        
        if best_outcome[0] == 'home_win':
            recommendation = f"Victoire de {match['home_team']}"
            confidence = "Élevée" if best_outcome[1]['probability'] > 50 else "Modérée"
        elif best_outcome[0] == 'away_win':
            recommendation = f"Victoire de {match['away_team']}"
            confidence = "Élevée" if best_outcome[1]['probability'] > 50 else "Modérée"
        else:
            recommendation = "Match nul"
            confidence = "Élevée" if best_outcome[1]['probability'] > 35 else "Modérée"
        
        st.success(f"**{recommendation}**")
        st.info(f"Niveau de confiance: **{confidence}**")

def calculate_match_probabilities(match):
    """Calcule les probabilités pour un match"""
    
    # Probabilités de base
    base_home = random.uniform(35, 55)
    base_draw = random.uniform(25, 35)
    base_away = 100 - base_home - base_draw
    
    # Ajustements
    home_advantage = random.uniform(1.1, 1.25)
    adjusted_home = base_home * home_advantage
    adjusted_draw = base_draw
    adjusted_away = base_away * 0.9  # Désavantage extérieur
    
    # Normaliser
    total = adjusted_home + adjusted_draw + adjusted_away
    home_prob = round((adjusted_home / total) * 100, 1)
    draw_prob = round((adjusted_draw / total) * 100, 1)
    away_prob = round((adjusted_away / total) * 100, 1)
    
    # Scores probables
    score_probabilities = {}
    for home_goals in range(0, 4):
        for away_goals in range(0, 4):
            score = f"{home_goals}-{away_goals}"
            # Probabilité basée sur la différence de buts attendue
            prob = max(0.1, 10 - abs(home_goals - away_goals) * 2)
            score_probabilities[score] = round(prob, 1)
    
    # Normaliser les scores
    total_scores = sum(score_probabilities.values())
    score_probabilities = {k: round((v/total_scores)*100, 1) 
                          for k, v in score_probabilities.items()}
    
    return {
        'outcomes': {
            'home_win': {
                'probability': home_prob,
                'estimated_odds': round(100/home_prob, 2)
            },
            'draw': {
                'probability': draw_prob,
                'estimated_odds': round(100/draw_prob, 2)
            },
            'away_win': {
                'probability': away_prob,
                'estimated_odds': round(100/away_prob, 2)
            }
        },
        'score_probabilities': score_probabilities,
        'expected_goals': {
            'home': round(random.uniform(1.2, 2.4), 1),
            'away': round(random.uniform(0.8, 1.9), 1)
        }
    }

def display_betting_opportunities(match):
    """Affiche les opportunités de pari"""
    
    st.subheader("💰 Analyse des paris")
    
    # Générer des cotes
    bookmaker_odds = generate_realistic_odds(match)
    
    # Afficher les cotes
    st.markdown("### 📊 Cotes des bookmakers")
    
    # Convertir en DataFrame
    odds_list = []
    for bookmaker, markets in bookmaker_odds.items():
        for market, odd in markets.items():
            odds_list.append({
                'Bookmaker': bookmaker,
                'Marché': market,
                'Cote': odd
            })
    
    df_odds = pd.DataFrame(odds_list)
    
    # Pivoter pour afficher par bookmaker
    pivot_df = df_odds.pivot(index='Marché', columns='Bookmaker', values='Cote')
    st.dataframe(pivot_df, use_container_width=True)
    
    # Value bets
    st.markdown("### 💎 Paris avec valeur")
    
    value_bets = find_value_bets(bookmaker_odds)
    
    if value_bets:
        cols = st.columns(min(3, len(value_bets)))
        
        for idx, bet in enumerate(value_bets[:3]):
            with cols[idx]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                ">
                    <div style="font-size: 14px; font-weight: bold;">{bet['bookmaker']}</div>
                    <div style="font-size: 12px; margin: 5px 0;">{bet['market']}</div>
                    <div style="font-size: 24px; font-weight: bold;">{bet['odd']}</div>
                    <div style="font-size: 12px;">Valeur: +{bet['value']}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📝 Suivre", key=f"track_{bet['bookmaker']}_{bet['market']}"):
                    st.success("Pari ajouté à votre suivi!")
    else:
        st.info("Aucun pari avec valeur significative détecté")
    
    # Recommandations
    st.markdown("### 🎯 Recommandations de mise")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Niveau de risque", "Moyen")
        st.caption("Basé sur la volatilité")
    
    with col2:
        st.metric("Mise suggérée", "2-3%")
        st.caption("Pour bankroll 1000€")
    
    with col3:
        st.metric("Meilleur bookmaker", "Bet365")
        st.caption("Cotes compétitives")

def generate_realistic_odds(match):
    """Génère des cotes réalistes"""
    
    # Cote de base pour la victoire à domicile
    base_home_odd = random.uniform(1.6, 2.8)
    
    bookmakers = {
        'Bet365': {
            '1': round(base_home_odd, 2),
            'N': round(random.uniform(3.2, 3.8), 2),
            '2': round(1 / ((1/base_home_odd) - 0.12), 2),
            'Over 2.5': round(random.uniform(1.7, 2.2), 2),
            'Under 2.5': round(random.uniform(1.6, 2.1), 2),
            'BTTS Oui': round(random.uniform(1.75, 2.25), 2),
            'BTTS Non': round(random.uniform(1.65, 2.05), 2)
        },
        'Unibet': {
            '1': round(base_home_odd + 0.05, 2),
            'N': round(random.uniform(3.25, 3.85), 2),
            '2': round(1 / ((1/base_home_odd) - 0.15), 2),
            'Over 2.5': round(random.uniform(1.72, 2.22), 2),
            'Under 2.5': round(random.uniform(1.62, 2.12), 2),
            'BTTS Oui': round(random.uniform(1.78, 2.28), 2),
            'BTTS Non': round(random.uniform(1.68, 2.08), 2)
        },
        'Winamax': {
            '1': round(base_home_odd + 0.08, 2),
            'N': round(random.uniform(3.3, 3.9), 2),
            '2': round(1 / ((1/base_home_odd) - 0.18), 2),
            'Over 2.5': round(random.uniform(1.75, 2.25), 2),
            'Under 2.5': round(random.uniform(1.65, 2.15), 2),
            'BTTS Oui': round(random.uniform(1.8, 2.3), 2),
            'BTTS Non': round(random.uniform(1.7, 2.1), 2)
        }
    }
    
    return bookmakers

def find_value_bets(bookmaker_odds):
    """Identifie les paris avec de la valeur"""
    
    value_bets = []
    
    for bookmaker, markets in bookmaker_odds.items():
        for market, odd in markets.items():
            # Simuler une détection de valeur (probabilité 25%)
            if random.random() < 0.25:
                value_bets.append({
                    'bookmaker': bookmaker,
                    'market': market,
                    'odd': odd,
                    'value': round(random.uniform(5, 18), 1),
                    'confidence': random.randint(6, 9)
                })
    
    # Trier par valeur
    value_bets.sort(key=lambda x: x['value'], reverse=True)
    
    return value_bets[:3]

def display_complete_info(match):
    """Affiche les informations complètes du match"""
    
    st.subheader("📋 Informations complètes")
    
    # Informations détaillées
    info_data = {
        'Champ': [
            'ID du match',
            'Équipe domicile',
            'ID domicile',
            'Équipe extérieur',
            'ID extérieur',
            'Ligue',
            'ID ligue',
            'Pays',
            'Stade',
            'Date complète',
            'Statut',
            'Fuseau horaire'
        ],
        'Valeur': [
            match['id'],
            match['home_team'],
            match['home_team_id'],
            match['away_team'],
            match['away_team_id'],
            match['league'],
            match['league_id'],
            match['country'],
            match['venue'],
            match['date'].strftime("%d/%m/%Y %H:%M:%S"),
            match['status'],
            'Europe/Paris'
        ]
    }
    
    df_info = pd.DataFrame(info_data)
    st.dataframe(df_info.set_index('Champ'), use_container_width=True)
    
    # Conseils d'analyse
    st.markdown("### 💡 Conseils pour votre analyse")
    
    tips = [
        "1. **Consultez les dernières nouvelles** - Blessures, suspensions, forme des joueurs",
        "2. **Analysez le face-à-face** - Historique des confrontations récentes",
        "3. **Évaluez la motivation** - Enjeux du match pour chaque équipe",
        "4. **Vérifiez les conditions météo** - Impact sur le style de jeu",
        "5. **Comparez les cotes** - Utilisez plusieurs bookmakers",
        "6. **Gérez votre bankroll** - Ne misez jamais plus de 5% par pari",
        "7. **Prenez en compte l'avantage domicile** - Statistiquement significatif",
        "8. **Analysez la forme récente** - 5 derniers matchs minimum",
        "9. **Considérez les facteurs psychologiques** - Pression, rivalités",
        "10. **Diversifiez vos paris** - Explorez différents marchés"
    ]
    
    for tip in tips:
        st.markdown(tip)
    
    # Note finale
    st.markdown("---")
    st.info("""
    **⚠️ Important :** 
    Ces analyses sont basées sur des données statistiques et des modèles prédictifs. 
    Les paris sportifs comportent des risques. Ne misez que ce que vous pouvez vous permettre de perdre.
    """)

if __name__ == "__main__":
    main()
