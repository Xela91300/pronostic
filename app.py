# app.py - Système de Paris Sportifs Professionnel
# Version Simplifiée pour Streamlit Cloud

import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import math

# =============================================================================
# CONFIGURATION DES API
# =============================================================================

class APIConfig:
    """Configuration des APIs"""
    
    # VOTRE clé API-Football
    API_FOOTBALL_KEY: str = "249b3051eCA063F0e381609128c00d7d"
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"
    
    # Configuration
    CACHE_TTL: int = 300
    REQUEST_TIMEOUT: int = 15

# =============================================================================
# CLIENT API
# =============================================================================

class FootballDataClient:
    """Client pour l'API Football"""
    
    def __init__(self):
        self.config = APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'x-apisports-key': self.config.API_FOOTBALL_KEY,
            'User-Agent': 'Mozilla/5.0'
        })
        self.cache = {}
        self.cache_timestamps = {}
    
    def test_connection(self) -> bool:
        """Teste la connexion à l'API"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/status"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_leagues(self) -> List[Dict]:
        """Récupère toutes les ligues disponibles"""
        cache_key = "all_leagues"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/leagues"
            response = self.session.get(url, timeout=self.config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                leagues = []
                
                for item in data:
                    league = item.get('league', {})
                    country = item.get('country', {})
                    
                    leagues.append({
                        'id': league.get('id'),
                        'name': league.get('name'),
                        'type': league.get('type'),
                        'logo': league.get('logo'),
                        'country': country.get('name'),
                        'country_code': country.get('code'),
                        'flag': country.get('flag'),
                        'season': item.get('seasons', [{}])[-1].get('year', 2024)
                    })
                
                self._cache_data(cache_key, leagues)
                return leagues
            
            return []
            
        except Exception as e:
            st.error(f"Erreur récupération ligues: {str(e)}")
            return []
    
    def get_fixtures(self, league_id: int, season: int = 2024, 
                    from_date: str = None, to_date: str = None, 
                    next: int = None) -> List[Dict]:
        """Récupère les matchs"""
        cache_key = f"fixtures_{league_id}_{season}_{from_date}_{to_date}_{next}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'league': league_id,
                'season': season,
                'timezone': 'Europe/Paris'
            }
            
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date
            if next:
                params['next'] = next
            
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                fixtures = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    goals = fixture.get('goals', {})
                    league = fixture.get('league', {})
                    
                    fixtures.append({
                        'fixture_id': fixture_data.get('id'),
                        'date': fixture_data.get('date'),
                        'timestamp': fixture_data.get('timestamp'),
                        'timezone': fixture_data.get('timezone'),
                        'venue': fixture_data.get('venue', {}).get('name'),
                        'status': fixture_data.get('status', {}),
                        'home_id': teams.get('home', {}).get('id'),
                        'home_name': teams.get('home', {}).get('name'),
                        'home_logo': teams.get('home', {}).get('logo'),
                        'away_id': teams.get('away', {}).get('id'),
                        'away_name': teams.get('away', {}).get('name'),
                        'away_logo': teams.get('away', {}).get('logo'),
                        'home_score': goals.get('home'),
                        'away_score': goals.get('away'),
                        'league_id': league.get('id'),
                        'league_name': league.get('name'),
                        'league_logo': league.get('logo'),
                        'league_country': league.get('country'),
                        'round': league.get('round')
                    })
                
                self._cache_data(cache_key, fixtures)
                return fixtures
            
            return []
            
        except Exception as e:
            st.error(f"Erreur récupération matchs: {str(e)}")
            return []
    
    def get_live_fixtures(self) -> List[Dict]:
        """Récupère les matchs en direct"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'live': 'all',
                'timezone': 'Europe/Paris'
            }
            
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                live_matches = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    goals = fixture.get('goals', {})
                    
                    live_match = {
                        'fixture_id': fixture_data.get('id'),
                        'date': fixture_data.get('date'),
                        'status': fixture_data.get('status', {}),
                        'elapsed': fixture_data.get('status', {}).get('elapsed', 0),
                        'home_id': teams.get('home', {}).get('id'),
                        'home_name': teams.get('home', {}).get('name'),
                        'home_logo': teams.get('home', {}).get('logo'),
                        'away_id': teams.get('away', {}).get('id'),
                        'away_name': teams.get('away', {}).get('name'),
                        'away_logo': teams.get('away', {}).get('logo'),
                        'home_score': goals.get('home', 0),
                        'away_score': goals.get('away', 0),
                        'is_live': True
                    }
                    
                    live_matches.append(live_match)
                
                return live_matches
            
            return []
            
        except Exception as e:
            st.warning(f"Erreur matchs en direct: {str(e)}")
            return []
    
    def get_team_statistics(self, team_id: int, league_id: int, season: int = 2024) -> Dict:
        """Récupère les statistiques d'une équipe"""
        cache_key = f"team_stats_{team_id}_{league_id}_{season}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams/statistics"
            params = {
                'team': team_id,
                'league': league_id,
                'season': season
            }
            
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json().get('response', {})
                self._cache_data(cache_key, data)
                return data
            
            return {}
            
        except Exception as e:
            st.warning(f"Statistiques équipe non disponibles: {str(e)}")
            return {}
    
    def get_fixture_statistics(self, fixture_id: int) -> List[Dict]:
        """Récupère les statistiques d'un match"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures/statistics"
            params = {'fixture': fixture_id}
            
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                return response.json().get('response', [])
            
            return []
            
        except Exception:
            return []
    
    def get_fixture_events(self, fixture_id: int) -> List[Dict]:
        """Récupère les événements d'un match"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures/events"
            params = {'fixture': fixture_id}
            
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                return response.json().get('response', [])
            
            return []
            
        except Exception:
            return []
    
    def get_team_fixtures(self, team_id: int, season: int = 2024, last: int = 10) -> List[Dict]:
        """Récupère les derniers matchs d'une équipe"""
        cache_key = f"team_fixtures_{team_id}_{season}_{last}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'team': team_id,
                'season': season,
                'last': last
            }
            
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                fixtures = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    goals = fixture.get('goals', {})
                    
                    fixtures.append({
                        'fixture_id': fixture_data.get('id'),
                        'date': fixture_data.get('date'),
                        'home_id': teams.get('home', {}).get('id'),
                        'home_name': teams.get('home', {}).get('name'),
                        'away_id': teams.get('away', {}).get('id'),
                        'away_name': teams.get('away', {}).get('name'),
                        'home_score': goals.get('home'),
                        'away_score': goals.get('away'),
                        'is_home': teams.get('home', {}).get('id') == team_id
                    })
                
                self._cache_data(cache_key, fixtures)
                return fixtures
            
            return []
            
        except Exception as e:
            st.warning(f"Historique équipe non disponible: {str(e)}")
            return []
    
    def get_odds(self, fixture_id: int, bookmaker: int = 6) -> Dict:
        """Récupère les cotes pour un match"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/odds"
            params = {
                'fixture': fixture_id,
                'bookmaker': bookmaker
            }
            
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                if data:
                    return data[0]
            
            return {}
            
        except Exception as e:
            st.warning(f"Cotes non disponibles: {str(e)}")
            return {}
    
    def _is_cached(self, key: str) -> bool:
        """Vérifie si les données sont en cache"""
        if key in self.cache and key in self.cache_timestamps:
            age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
            return age < self.config.CACHE_TTL
        return False
    
    def _cache_data(self, key: str, data):
        """Met en cache les données"""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()

# =============================================================================
# SYSTÈME ELO SIMPLIFIÉ
# =============================================================================

class EloSystem:
    """Système Elo simplifié pour le football"""
    
    def __init__(self, k_factor: float = 32.0, home_advantage: float = 70.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.team_ratings = {}
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calcule le score attendu entre deux ratings"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_id: int, away_id: int, 
                      home_score: int, away_score: int):
        """Met à jour les ratings après un match"""
        
        # Initialiser les équipes si nécessaire
        if home_id not in self.team_ratings:
            self.team_ratings[home_id] = 1500
        if away_id not in self.team_ratings:
            self.team_ratings[away_id] = 1500
        
        home_rating = self.team_ratings[home_id]
        away_rating = self.team_ratings[away_id]
        
        # Déterminer le résultat
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score < away_score:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5
        
        # Score attendu avec avantage domicile
        expected_home = self.calculate_expected_score(home_rating + self.home_advantage, away_rating)
        expected_away = 1 - expected_home
        
        # Mise à jour Elo
        home_elo_change = self.k_factor * (actual_home - expected_home)
        away_elo_change = self.k_factor * (actual_away - expected_away)
        
        self.team_ratings[home_id] += home_elo_change
        self.team_ratings[away_id] += away_elo_change
        
        return home_elo_change
    
    def predict_match(self, home_id: int, away_id: int) -> Dict:
        """Prédit les probabilités d'un match"""
        
        # Ratings
        home_elo = self.team_ratings.get(home_id, 1500)
        away_elo = self.team_ratings.get(away_id, 1500)
        
        # Probabilité victoire domicile avec avantage
        home_win_prob = self.calculate_expected_score(home_elo + self.home_advantage, away_elo)
        
        # Probabilité match nul
        elo_diff = abs((home_elo + self.home_advantage) - away_elo)
        draw_prob = 0.25 * (1 - elo_diff / 800)
        draw_prob = max(0.1, min(draw_prob, 0.35))
        
        # Probabilité victoire extérieur
        away_win_prob = 1 - home_win_prob - draw_prob
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'home_elo': home_elo,
            'away_elo': away_elo
        }
    
    def train_from_fixtures(self, fixtures: List[Dict]):
        """Entraîne le système Elo sur des matchs historiques"""
        for fixture in fixtures:
            home_id = fixture.get('home_id')
            away_id = fixture.get('away_id')
            home_score = fixture.get('home_score')
            away_score = fixture.get('away_score')
            
            if all([home_id, away_id, home_score is not None, away_score is not None]):
                self.update_ratings(home_id, away_id, home_score, away_score)

# =============================================================================
# VALUE BET DETECTOR
# =============================================================================

class ValueBetDetector:
    """Détecteur de value bets simplifié"""
    
    def __init__(self, elo_system: EloSystem):
        self.elo = elo_system
        self.min_edge = 0.02
    
    def analyze_fixture(self, fixture: Dict, odds_data: Dict = None) -> Optional[Dict]:
        """Analyse un match pour détecter les value bets"""
        try:
            home_id = fixture.get('home_id')
            away_id = fixture.get('away_id')
            home_name = fixture.get('home_name')
            away_name = fixture.get('away_name')
            
            if not all([home_id, away_id, home_name, away_name]):
                return None
            
            # Prédiction Elo
            prediction = self.elo.predict_match(home_id, away_id)
            
            # Si pas de cotes, retourner juste la prédiction
            if not odds_data:
                return {
                    'fixture_id': fixture.get('fixture_id'),
                    'match': f"{home_name} vs {away_name}",
                    'league': fixture.get('league_name'),
                    'date': fixture.get('date'),
                    'prediction': prediction,
                    'value_bets': [],
                    'status': 'no_odds'
                }
            
            # Extraire les cotes
            bookmaker_odds = self._extract_odds(odds_data)
            
            # Analyser chaque marché
            value_bets = []
            
            # Marché 1X2
            h2h_analysis = self._analyze_h2h(prediction, bookmaker_odds)
            if h2h_analysis:
                value_bets.extend(h2h_analysis)
            
            # Filtrer les value bets significatifs
            significant_bets = [bet for bet in value_bets if bet['edge'] >= self.min_edge]
            
            return {
                'fixture_id': fixture.get('fixture_id'),
                'match': f"{home_name} vs {away_name}",
                'league': fixture.get('league_name'),
                'date': fixture.get('date'),
                'prediction': prediction,
                'value_bets': significant_bets,
                'odds_available': bool(bookmaker_odds),
                'status': 'analyzed'
            }
            
        except Exception as e:
            st.error(f"Erreur analyse match: {str(e)}")
            return None
    
    def _extract_odds(self, odds_data: Dict) -> Dict:
        """Extrait les cotes de l'API"""
        try:
            bookmakers = odds_data.get('bookmakers', [])
            if not bookmakers:
                return {}
            
            # Prendre le premier bookmaker
            bookmaker = bookmakers[0]
            bets = {}
            
            for bet in bookmaker.get('bets', []):
                market = bet.get('name', '').lower()
                
                if market == 'match winner':
                    for value in bet.get('values', []):
                        outcome = value.get('value', '').lower()
                        odd = float(value.get('odd', 1))
                        
                        if 'home' in outcome:
                            bets['home'] = odd
                        elif 'draw' in outcome:
                            bets['draw'] = odd
                        elif 'away' in outcome:
                            bets['away'] = odd
            
            return bets
            
        except Exception:
            return {}
    
    def _analyze_h2h(self, prediction: Dict, odds: Dict) -> List[Dict]:
        """Analyse le marché 1X2"""
        value_bets = []
        
        markets = [
            {'key': 'home', 'prob': prediction['home_win'], 'name': '1'},
            {'key': 'draw', 'prob': prediction['draw'], 'name': 'X'},
            {'key': 'away', 'prob': prediction['away_win'], 'name': '2'}
        ]
        
        for market in markets:
            odds_value = odds.get(market['key'])
            
            if odds_value and odds_value > 1:
                edge = self._calculate_edge(market['prob'], odds_value)
                
                if edge >= self.min_edge:
                    value_bet = {
                        'market': '1X2',
                        'selection': market['name'],
                        'probability': market['prob'],
                        'odds': odds_value,
                        'edge': edge,
                        'implied_prob': 1 / odds_value,
                        'value_score': edge * market['prob'] * 100
                    }
                    
                    value_bets.append(value_bet)
        
        return value_bets
    
    def _calculate_edge(self, probability: float, odds: float) -> float:
        """Calcule l'edge (valeur attendue)"""
        if odds <= 1 or probability <= 0:
            return -1
        return (probability * odds) - 1

# =============================================================================
# BET MANAGER
# =============================================================================

class BetManager:
    """Gestionnaire de bankroll et paris"""
    
    def __init__(self, initial_bankroll: float = 10000.0):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.bets = []
        self.performance = {
            'total_bets': 0,
            'won': 0,
            'lost': 0,
            'pending': 0,
            'total_staked': 0.0,
            'total_return': 0.0,
            'total_profit': 0.0,
            'roi': 0.0,
            'win_rate': 0.0
        }
    
    def place_bet(self, match_info: Dict, bet_details: Dict, 
                  stake: float, odds: float) -> Dict:
        """Place un pari"""
        if stake > self.bankroll:
            return {'success': False, 'error': 'Bankroll insuffisant'}
        
        bet_id = len(self.bets) + 1
        bet = {
            'id': bet_id,
            'timestamp': datetime.now(),
            'match': match_info.get('match'),
            'league': match_info.get('league'),
            'market': bet_details.get('market'),
            'selection': bet_details.get('selection'),
            'stake': stake,
            'odds': odds,
            'probability': bet_details.get('probability'),
            'edge': bet_details.get('edge'),
            'potential_win': stake * (odds - 1),
            'potential_return': stake * odds,
            'status': 'pending',
            'result': None,
            'settled_at': None
        }
        
        self.bankroll -= stake
        self.bets.append(bet)
        
        # Mettre à jour les stats
        self.performance['total_bets'] += 1
        self.performance['pending'] += 1
        self.performance['total_staked'] += stake
        
        return {'success': True, 'bet': bet, 'bankroll': self.bankroll}
    
    def settle_bet(self, bet_id: int, result: str) -> Dict:
        """Règle un pari"""
        bet = next((b for b in self.bets if b['id'] == bet_id), None)
        
        if not bet:
            return {'success': False, 'error': 'Pari non trouvé'}
        
        bet['status'] = 'settled'
        bet['result'] = result
        bet['settled_at'] = datetime.now()
        
        self.performance['pending'] -= 1
        
        if result == 'win':
            winnings = bet['stake'] * bet['odds']
            self.bankroll += winnings
            self.performance['won'] += 1
            self.performance['total_return'] += winnings
            self.performance['total_profit'] += (winnings - bet['stake'])
        
        elif result == 'loss':
            self.performance['lost'] += 1
        
        elif result == 'void':
            # Remboursement
            self.bankroll += bet['stake']
            self.performance['total_staked'] -= bet['stake']
        
        # Recalculer les métriques
        self._update_performance()
        
        return {'success': True, 'bet': bet, 'bankroll': self.bankroll}
    
    def _update_performance(self):
        """Met à jour les métriques de performance"""
        if self.performance['total_bets'] > 0:
            total_settled = self.performance['won'] + self.performance['lost']
            if total_settled > 0:
                self.performance['win_rate'] = (self.performance['won'] / total_settled) * 100
            
            if self.performance['total_staked'] > 0:
                self.performance['roi'] = (self.performance['total_profit'] / 
                                         self.performance['total_staked']) * 100

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def setup_interface():
    """Configure l'interface Streamlit"""
    st.set_page_config(
        page_title="Système de Paris Sportifs Pro",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .match-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1E88E5, #0D47A1);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .profit-positive { color: #4CAF50; font-weight: bold; }
    .profit-negative { color: #f44336; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">💰 SYSTÈME DE PARIS FOOTBALL PRO</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyse de matchs • Value Bets • Gestion Bankroll</div>', unsafe_allow_html=True)

def main():
    """Application principale"""
    setup_interface()
    
    # Initialisation des composants
    if 'api_client' not in st.session_state:
        st.session_state.api_client = FootballDataClient()
    
    if 'elo_system' not in st.session_state:
        st.session_state.elo_system = EloSystem()
    
    if 'value_detector' not in st.session_state:
        st.session_state.value_detector = ValueBetDetector(st.session_state.elo_system)
    
    if 'bet_manager' not in st.session_state:
        st.session_state.bet_manager = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        # Test connexion API
        if st.button("🔗 Tester la connexion API"):
            if st.session_state.api_client.test_connection():
                st.success("✅ Connexion API réussie !")
            else:
                st.error("❌ Échec connexion API")
        
        # Bankroll
        st.subheader("💰 BANKROLL")
        
        if st.session_state.bet_manager is None:
            initial_bankroll = st.number_input(
                "Bankroll initial (€)",
                min_value=100.0,
                max_value=1000000.0,
                value=10000.0,
                step=500.0
            )
            
            if st.button("Initialiser le bankroll", type="primary"):
                st.session_state.bet_manager = BetManager(initial_bankroll)
                st.success(f"Bankroll initialisé: €{initial_bankroll:,.2f}")
                st.rerun()
        else:
            st.metric("Bankroll actuel", f"€{st.session_state.bet_manager.bankroll:,.2f}")
            st.metric("Profit total", f"€{st.session_state.bet_manager.performance['total_profit']:,.2f}")
        
        # Paramètres
        st.subheader("🎯 PARAMÈTRES")
        
        min_edge = st.slider("Edge minimum (%)", 1.0, 10.0, 2.0, 0.5)
        st.session_state.value_detector.min_edge = min_edge / 100
        
        # Info
        st.divider()
        st.info(f"""
        **📊 VOTRE SYSTÈME**
        - Clé API: ✅ Active
        - Mode: Professionnel
        - Dernière MAJ: {datetime.now().strftime('%H:%M:%S')}
        """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Dashboard", 
        "🎯 Value Bets", 
        "⚽ Matchs en Direct", 
        "💰 Paris", 
        "📈 Performances",
        "🔍 Analyse Match"
    ])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        display_value_bets()
    
    with tab3:
        display_live_matches()
    
    with tab4:
        display_betting_interface()
    
    with tab5:
        display_performance()
    
    with tab6:
        display_match_analysis_manual()

def display_dashboard():
    """Affiche le dashboard principal"""
    st.header("📊 DASHBOARD")
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.bet_manager:
            bankroll = st.session_state.bet_manager.bankroll
            initial = st.session_state.bet_manager.initial_bankroll
            profit = bankroll - initial
            delta = f"€{profit:+,.0f}"
            st.metric("💰 BANKROLL", f"€{bankroll:,.0f}", delta)
        else:
            st.metric("💰 BANKROLL", "€10,000", "Non initialisé")
    
    with col2:
        if st.session_state.bet_manager:
            roi = st.session_state.bet_manager.performance['roi']
            st.metric("📈 ROI", f"{roi:.1f}%")
        else:
            st.metric("📈 ROI", "0.0%")
    
    with col3:
        if st.session_state.bet_manager:
            win_rate = st.session_state.bet_manager.performance['win_rate']
            st.metric("🎯 RÉUSSITE", f"{win_rate:.1f}%")
        else:
            st.metric("🎯 RÉUSSITE", "0.0%")
    
    with col4:
        # Test connexion API
        if st.session_state.api_client.test_connection():
            st.metric("🌐 API", "✅ Connectée")
        else:
            st.metric("🌐 API", "❌ Déconnectée")
    
    # Boutons d'action
    st.subheader("🚀 ACTIONS RAPIDES")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Scanner value bets", type="primary"):
            st.session_state.scan_value_bets = True
            st.info("Allez dans l'onglet '🎯 Value Bets'")
    
    with col2:
        if st.button("🔄 Rafraîchir données"):
            st.session_state.api_client.cache.clear()
            st.success("Cache vidé !")
    
    with col3:
        if st.button("📊 Entraîner modèle"):
            with st.spinner("Entraînement en cours..."):
                # Charger des matchs historiques
                leagues = st.session_state.api_client.get_leagues()
                if leagues:
                    premier_league = next((l for l in leagues if "Premier League" in l['name']), None)
                    if premier_league:
                        # Utiliser les matchs des 7 derniers jours
                        from_date = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')
                        to_date = date.today().strftime('%Y-%m-%d')
                        
                        fixtures = st.session_state.api_client.get_fixtures(
                            premier_league['id'],
                            premier_league['season'],
                            from_date=from_date,
                            to_date=to_date
                        )
                        st.session_state.elo_system.train_from_fixtures(fixtures)
                        st.success(f"Modèle entraîné sur {len(fixtures)} matchs !")
    
    # Aperçu des matchs du jour
    st.subheader("📅 MATCHS DU JOUR")
    
    try:
        leagues = st.session_state.api_client.get_leagues()
        if leagues:
            premier_league = next((l for l in leagues if "Premier League" in l['name']), None)
            if premier_league:
                today = date.today().strftime('%Y-%m-%d')
                fixtures = st.session_state.api_client.get_fixtures(
                    premier_league['id'],
                    premier_league['season'],
                    from_date=today,
                    to_date=today
                )
                
                if fixtures:
                    for fixture in fixtures[:5]:  # Afficher 5 matchs max
                        st.write(f"**{fixture['home_name']} vs {fixture['away_name']}**")
                        st.write(f"📍 {fixture['venue']} • {fixture['date'][11:16]}")
                        st.divider()
                else:
                    st.info("Aucun match prévu aujourd'hui dans cette ligue.")
    except:
        st.info("Chargement des matchs du jour...")

def display_value_bets():
    """Affiche les value bets détectés"""
    st.header("🎯 VALUE BETS")
    
    # Sélection de la ligue
    st.subheader("1. Sélectionnez une ligue")
    
    leagues = st.session_state.api_client.get_leagues()
    
    if not leagues:
        st.error("Impossible de charger les ligues. Vérifiez votre connexion API.")
        return
    
    # Filtrer les ligues majeures
    major_leagues = [
        "Premier League", "La Liga", "Serie A", 
        "Bundesliga", "Ligue 1", "Champions League"
    ]
    
    filtered_leagues = [l for l in leagues if any(ml in l['name'] for ml in major_leagues)]
    
    if not filtered_leagues:
        st.warning("Aucune ligue majeure trouvée.")
        return
    
    league_options = {f"{l['name']} ({l['country']})": l for l in filtered_leagues}
    
    selected_league_name = st.selectbox(
        "Choisir une ligue",
        list(league_options.keys()),
        index=0
    )
    
    if selected_league_name:
        selected_league = league_options[selected_league_name]
        
        # Période
        col1, col2 = st.columns(2)
        with col1:
            days_ahead = st.slider("Jours à venir", 1, 14, 3)
        with col2:
            max_matches = st.slider("Max matchs", 10, 50, 20)
        
        # Scanner les value bets
        if st.button("🔍 Scanner cette ligue", type="primary"):
            with st.spinner(f"Analyse de {selected_league['name']}..."):
                try:
                    # Récupérer les matchs
                    from_date = date.today().strftime('%Y-%m-%d')
                    to_date = (date.today() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                    
                    fixtures = st.session_state.api_client.get_fixtures(
                        selected_league['id'],
                        selected_league['season'],
                        from_date=from_date,
                        to_date=to_date
                    )
                    
                    if not fixtures:
                        st.warning(f"Aucun match trouvé pour {selected_league['name']}")
                        return
                    
                    # Limiter le nombre
                    fixtures = fixtures[:max_matches]
                    
                    # Analyser chaque match
                    value_bets_found = []
                    
                    for fixture in fixtures:
                        # Récupérer les cotes
                        odds_data = st.session_state.api_client.get_odds(fixture['fixture_id'])
                        
                        # Analyser
                        analysis = st.session_state.value_detector.analyze_fixture(fixture, odds_data)
                        
                        if analysis and analysis['value_bets']:
                            value_bets_found.append(analysis)
                    
                    # Afficher les résultats
                    if value_bets_found:
                        st.success(f"✅ {len(value_bets_found)} value bets détectés !")
                        
                        for analysis in value_bets_found:
                            with st.expander(f"🎯 {analysis['match']}"):
                                display_match_analysis(analysis)
                    
                    else:
                        st.info("Aucun value bet détecté avec les paramètres actuels.")
                        
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {str(e)}")

def display_match_analysis(analysis: Dict):
    """Affiche l'analyse d'un match"""
    st.write(f"**Ligue:** {analysis['league']}")
    st.write(f"**Date:** {analysis['date'][:10]} {analysis['date'][11:16]}")
    
    # Prédiction
    pred = analysis['prediction']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("1", f"{pred['home_win']*100:.1f}%")
    
    with col2:
        st.metric("N", f"{pred['draw']*100:.1f}%")
    
    with col3:
        st.metric("2", f"{pred['away_win']*100:.1f}%")
    
    # Value bets
    st.subheader("💰 Value Bets")
    
    for value_bet in analysis['value_bets']:
        st.markdown(f"""
        <div class="value-bet-card">
            <h4>🎯 {value_bet['market']} - {value_bet['selection']}</h4>
            <p><strong>Edge:</strong> {value_bet['edge']*100:.2f}% • 
            <strong>Cote:</strong> {value_bet['odds']:.2f}</p>
            <p>Probabilité modèle: {value_bet['probability']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton pour placer le pari
        if st.session_state.bet_manager:
            if st.button(f"📝 Parier {value_bet['selection']}", key=f"bet_{analysis['fixture_id']}_{value_bet['selection']}"):
                
                match_info = {
                    'match': analysis['match'],
                    'league': analysis['league']
                }
                
                bet_details = {
                    'market': value_bet['market'],
                    'selection': value_bet['selection'],
                    'probability': value_bet['probability'],
                    'edge': value_bet['edge']
                }
                
                # Calcul de mise simple
                bankroll = st.session_state.bet_manager.bankroll
                stake = bankroll * 0.02  # 2% du bankroll
                
                result = st.session_state.bet_manager.place_bet(
                    match_info, bet_details, stake, value_bet['odds']
                )
                
                if result['success']:
                    st.success(f"✅ Pari placé ! Mise: €{stake:,.2f}")
                else:
                    st.error(f"❌ Erreur: {result.get('error')}")

def display_live_matches():
    """Affiche les matchs en direct"""
    st.header("⚽ MATCHS EN DIRECT")
    
    if st.button("🔄 Actualiser"):
        st.session_state.api_client.cache.clear()
    
    with st.spinner("Chargement des matchs en direct..."):
        live_matches = st.session_state.api_client.get_live_fixtures()
        
        if not live_matches:
            st.info("Aucun match en cours actuellement.")
            return
        
        st.success(f"📡 {len(live_matches)} match(s) en direct")
        
        for match in live_matches:
            with st.expander(f"🔥 {match['home_name']} {match['home_score']} - {match['away_score']} {match['away_name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Statut:** {match['status'].get('long', 'En cours')}")
                    st.write(f"**Temps:** {match['elapsed']}'")
                    
                    # Événements
                    events = st.session_state.api_client.get_fixture_events(match['fixture_id'])
                    if events:
                        st.write("**Événements:**")
                        for event in events[-3:]:
                            time = event.get('time', {}).get('elapsed', '')
                            type_ = event.get('type', '')
                            player = event.get('player', {}).get('name', '')
                            
                            if type_ == 'Goal':
                                st.write(f"⚽ {time}' - {player}")
                
                with col2:
                    # Statistiques
                    stats = st.session_state.api_client.get_fixture_statistics(match['fixture_id'])
                    if stats:
                        st.write("**Statistiques:**")
                        for team_stats in stats[:2]:
                            team_name = team_stats.get('team', {}).get('name', '')
                            st.write(f"**{team_name}:**")
                            
                            for stat in team_stats.get('statistics', [])[:2]:
                                st.write(f"{stat.get('type')}: {stat.get('value')}")

def display_betting_interface():
    """Interface de placement de paris"""
    st.header("💰 INTERFACE DE PARIS")
    
    if st.session_state.bet_manager is None:
        st.warning("Veuillez d'abord initialiser le bankroll dans la sidebar.")
        return
    
    # Interface de pari manuel
    st.subheader("📝 Pari Manuel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        match_name = st.text_input("Match", "Manchester City vs Liverpool")
        league = st.text_input("Ligue", "Premier League")
        market = st.selectbox("Marché", ["1X2", "Double Chance", "Over/Under 2.5"])
        selection = st.selectbox("Sélection", ["1", "X", "2", "Over", "Under"])
    
    with col2:
        odds = st.number_input("Cote", min_value=1.01, max_value=100.0, value=2.0, step=0.01)
        
        bankroll = st.session_state.bet_manager.bankroll
        stake_method = st.selectbox("Méthode de mise", 
                                   ["Montant fixe", "% du bankroll"])
        
        if stake_method == "Montant fixe":
            stake = st.number_input("Mise (€)", min_value=1.0, 
                                   max_value=bankroll, 
                                   value=100.0, step=10.0)
        else:
            percent = st.slider("Pourcentage", 0.5, 10.0, 2.0, 0.5)
            stake = bankroll * (percent / 100)
            st.write(f"**Mise:** €{stake:,.2f} ({percent}%)")
    
    potential_return = stake * odds
    potential_profit = potential_return - stake
    
    st.metric("💰 Retour potentiel", f"€{potential_return:,.2f}")
    st.metric("📈 Profit potentiel", f"€{potential_profit:,.2f}")
    
    if st.button("✅ Placer le pari", type="primary"):
        match_info = {
            'match': match_name,
            'league': league
        }
        
        bet_details = {
            'market': market,
            'selection': selection,
            'probability': 0.5,
            'edge': 0
        }
        
        result = st.session_state.bet_manager.place_bet(match_info, bet_details, stake, odds)
        
        if result['success']:
            st.success(f"""
            ✅ Pari placé avec succès !
            - ID: {result['bet']['id']}
            - Mise: €{stake:,.2f}
            - Bankroll restant: €{result['bankroll']:,.2f}
            """)
            st.rerun()
        else:
            st.error(f"❌ Erreur: {result.get('error')}")
    
    # Paris ouverts
    st.subheader("📋 Paris en cours")
    
    open_bets = [b for b in st.session_state.bet_manager.bets if b['status'] == 'pending']
    
    if open_bets:
        for bet in open_bets:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.write(f"**{bet['match']}**")
                    st.write(f"{bet['market']} - {bet['selection']} @ {bet['odds']:.2f}")
                with col2:
                    st.write(f"**Mise:** €{bet['stake']:,.2f}")
                with col3:
                    # Options pour régler le pari
                    result_col1, result_col2 = st.columns(2)
                    with result_col1:
                        if st.button("✅ Gagné", key=f"win_{bet['id']}"):
                            st.session_state.bet_manager.settle_bet(bet['id'], 'win')
                            st.rerun()
                    with result_col2:
                        if st.button("❌ Perdu", key=f"loss_{bet['id']}"):
                            st.session_state.bet_manager.settle_bet(bet['id'], 'loss')
                            st.rerun()
                st.divider()
    else:
        st.info("Aucun pari en cours.")

def display_performance():
    """Affiche les performances"""
    st.header("📈 PERFORMANCES")
    
    if st.session_state.bet_manager is None:
        st.warning("Veuillez initialiser le bankroll pour voir les performances.")
        return
    
    perf = st.session_state.bet_manager.performance
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Paris", perf['total_bets'])
        st.metric("En cours", perf['pending'])
    
    with col2:
        st.metric("Gagnés", perf['won'])
        st.metric("Perdus", perf['lost'])
    
    with col3:
        win_rate = perf['win_rate']
        st.metric("Taux Réussite", f"{win_rate:.1f}%")
        st.metric("ROI", f"{perf['roi']:.1f}%")
    
    with col4:
        st.metric("Mise Totale", f"€{perf['total_staked']:,.0f}")
        st.metric("Profit Total", f"€{perf['total_profit']:,.0f}")
    
    # Détails des paris
    st.subheader("📋 Historique détaillé")
    
    if st.session_state.bet_manager.bets:
        history_data = []
        
        for bet in st.session_state.bet_manager.bets:
            profit = ""
            if bet['status'] == 'settled':
                if bet['result'] == 'win':
                    profit = f"€{(bet['stake'] * bet['odds']) - bet['stake']:,.2f}"
                elif bet['result'] == 'loss':
                    profit = f"-€{bet['stake']:,.2f}"
            
            history_data.append({
                'ID': bet['id'],
                'Date': bet['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Match': bet['match'],
                'Marché': bet['market'],
                'Sélection': bet['selection'],
                'Cote': f"{bet['odds']:.2f}",
                'Mise': f"€{bet['stake']:,.2f}",
                'Statut': bet['status'],
                'Résultat': bet.get('result', 'N/A'),
                'Profit': profit
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun pari dans l'historique.")

def display_match_analysis_manual():
    """Analyse manuelle d'un match"""
    st.header("🔍 ANALYSE DE MATCH MANUELLE")
    
    st.info("Entrez les détails d'un match pour obtenir une analyse détaillée.")
    
    # Formulaire de saisie
    with st.form("match_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Équipe Domicile")
            home_team = st.text_input("Nom équipe domicile", "Manchester City")
            home_form = st.slider("Forme (1-10)", 1, 10, 7)
            home_attack = st.number_input("Attaque", 0.0, 5.0, 2.3, 0.1)
            home_defense = st.number_input("Défense", 0.0, 5.0, 0.8, 0.1)
        
        with col2:
            st.subheader("⚽ Équipe Extérieur")
            away_team = st.text_input("Nom équipe extérieur", "Liverpool")
            away_form = st.slider("Forme (1-10)", 1, 10, 6)
            away_attack = st.number_input("Attaque", 0.0, 5.0, 1.9, 0.1)
            away_defense = st.number_input("Défense", 0.0, 5.0, 1.2, 0.1)
        
        col3, col4 = st.columns(2)
        with col3:
            is_neutral = st.checkbox("Terrain neutre")
            importance = st.selectbox("Importance", ["Normal", "Coupe", "Dernière journée", "Finale"])
        
        with col4:
            weather = st.selectbox("Météo", ["Bonnes", "Pluie", "Vent", "Froid", "Chaud"])
            home_missing = st.number_input("Absents domicile", 0, 10, 1)
            away_missing = st.number_input("Absents extérieur", 0, 10, 2)
        
        submitted = st.form_submit_button("🚀 ANALYSER LE MATCH", type="primary")
    
    if submitted:
        try:
            # 1. CALCUL DES RATINGS
            st.subheader("📈 RATINGS DES ÉQUIPES")
            
            # Calcul simplifié
            home_rating = 1500 + (home_form - 5) * 50
            away_rating = 1500 + (away_form - 5) * 50
            
            if not is_neutral:
                home_rating += 70
            
            if importance in ["Finale", "Dernière journée"]:
                home_rating *= 1.1
                away_rating *= 1.1
            
            col5, col6 = st.columns(2)
            
            with col5:
                st.metric(f"🏠 {home_team}", f"{home_rating:.0f}")
                st.write(f"**Forme:** {home_form}/10")
                st.write(f"**Attaque:** {home_attack}")
                st.write(f"**Défense:** {home_defense}")
            
            with col6:
                st.metric(f"⚽ {away_team}", f"{away_rating:.0f}")
                st.write(f"**Forme:** {away_form}/10")
                st.write(f"**Attaque:** {away_attack}")
                st.write(f"**Défense:** {away_defense}")
            
            # 2. PRÉDICTIONS
            st.subheader("🎯 PRÉDICTIONS")
            
            rating_diff = home_rating - away_rating
            home_win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
            draw_prob = 0.25 * (1 - abs(home_win_prob - 0.5) * 2)
            away_win_prob = 1 - home_win_prob - draw_prob
            
            # Affichage
            col7, col8, col9 = st.columns(3)
            
            with col7:
                st.markdown(f"""
                <div style="background: #E3F2FD; padding: 15px; border-radius: 10px; text-align: center;">
                <h3>🏠 VICTOIRE</h3>
                <h2>{home_win_prob*100:.1f}%</h2>
                <p>Cote: {1/home_win_prob:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                st.markdown(f"""
                <div style="background: #F3E5F5; padding: 15px; border-radius: 10px; text-align: center;">
                <h3>🤝 NUL</h3>
                <h2>{draw_prob*100:.1f}%</h2>
                <p>Cote: {1/draw_prob:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col9:
                st.markdown(f"""
                <div style="background: #E8F5E9; padding: 15px; border-radius: 10px; text-align: center;">
                <h3>⚽ VICTOIRE</h3>
                <h2>{away_win_prob*100:.1f}%</h2>
                <p>Cote: {1/away_win_prob:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 3. SCORE PRÉDIT
            st.subheader("📊 SCORE ATTENDU")
            
            expected_home = (home_attack + away_defense) / 2
            expected_away = (away_attack + home_defense) / 2
            
            if weather != "Bonnes":
                expected_home *= 0.9
                expected_away *= 0.9
            
            predicted_home = round(expected_home)
            predicted_away = round(expected_away)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 25px; border-radius: 10px; text-align: center; color: white;">
            <h1 style="font-size: 3.5rem;">{predicted_home} - {predicted_away}</h1>
            <p>Score le plus probable</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. RECOMMANDATIONS
            st.subheader("💰 RECOMMANDATIONS")
            
            market_home = 1/home_win_prob * 0.9
            market_draw = 1/draw_prob * 0.9
            market_away = 1/away_win_prob * 0.9
            
            recommendations = []
            
            edge_home = (home_win_prob * market_home) - 1
            if edge_home > 0.02:
                recommendations.append({
                    'sélection': f"{home_team} (1)",
                    'cote': f"{market_home:.2f}",
                    'edge': f"{edge_home*100:.1f}%"
                })
            
            edge_draw = (draw_prob * market_draw) - 1
            if edge_draw > 0.02:
                recommendations.append({
                    'sélection': "Match Nul (X)",
                    'cote': f"{market_draw:.2f}",
                    'edge': f"{edge_draw*100:.1f}%"
                })
            
            edge_away = (away_win_prob * market_away) - 1
            if edge_away > 0.02:
                recommendations.append({
                    'sélection': f"{away_team} (2)",
                    'cote': f"{market_away:.2f}",
                    'edge': f"{edge_away*100:.1f}%"
                })
            
            if recommendations:
                st.success(f"✅ {len(recommendations)} opportunité(s)")
                
                for rec in recommendations:
                    with st.expander(rec['sélection']):
                        st.write(f"**Cote estimée:** {rec['cote']}")
                        st.write(f"**Edge:** {rec['edge']}")
            else:
                st.warning("⚠️ Aucune opportunité significative")
            
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == "__main__":
    main()
