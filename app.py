# app.py - Système de Paris Sportifs en Temps Réel
# Version Professionnelle avec VOTRE API intégrée

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
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import math

# =============================================================================
# CONFIGURATION DES API
# =============================================================================

class APIConfig:
    """Configuration des APIs avec VOTRE clé"""
    
    # VOTRE clé API-Football
    API_FOOTBALL_KEY: str = "249b3051eCA063F0e381609128c00d7d"
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"
    
    # The Odds API (optionnel - vous pouvez l'ajouter plus tard)
    ODDS_API_KEY: str = st.secrets.get("ODDS_API_KEY", "")
    ODDS_API_URL: str = "https://api.the-odds-api.com/v4"
    
    # Configuration
    CACHE_TTL: int = 300  # 5 minutes cache
    REQUEST_TIMEOUT: int = 15

# =============================================================================
# CLIENT API PROFESSIONNEL
# =============================================================================

class FootballDataClient:
    """Client pour l'API Football avec votre clé"""
    
    def __init__(self):
        self.config = APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'x-apisports-key': self.config.API_FOOTBALL_KEY,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
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
                    events = fixture.get('events', [])
                    statistics = fixture.get('statistics', [])
                    
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
                        'events': events,
                        'statistics': statistics,
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
# SYSTÈME ELO AVANCÉ
# =============================================================================

class AdvancedEloSystem:
    """Système Elo avancé pour le football"""
    
    def __init__(self, k_factor: float = 32.0, home_advantage: float = 70.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.team_ratings = {}
        self.team_history = {}
    
    def calculate_expected_score(self, rating_a: float, rating_b: float, 
                               home_advantage: float = 0) -> float:
        """Calcule le score attendu entre deux ratings"""
        return 1 / (1 + 10 ** ((rating_b - rating_a - home_advantage) / 400))
    
    def margin_of_victory_multiplier(self, goal_diff: int, rating_diff: float) -> float:
        """Multiplicateur basé sur la marge de victoire"""
        return np.log(abs(goal_diff) + 1) * (2.2 / (abs(rating_diff) * 0.001 + 2.2))
    
    def update_ratings(self, home_id: int, away_id: int, 
                      home_score: int, away_score: int, 
                      match_date: datetime = None, is_neutral: bool = False):
        """Met à jour les ratings après un match"""
        
        # Initialiser les équipes si nécessaire
        if home_id not in self.team_ratings:
            self.team_ratings[home_id] = {'elo': 1500, 'games': 0, 'last_update': match_date}
        if away_id not in self.team_ratings:
            self.team_ratings[away_id] = {'elo': 1500, 'games': 0, 'last_update': match_date}
        
        home_rating = self.team_ratings[home_id]['elo']
        away_rating = self.team_ratings[away_id]['elo']
        
        # Appliquer décay temporel
        self.apply_time_decay(home_id, match_date)
        self.apply_time_decay(away_id, match_date)
        
        home_rating = self.team_ratings[home_id]['elo']
        away_rating = self.team_ratings[away_id]['elo']
        
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
        
        # Avantage terrain
        home_adv = 0 if is_neutral else self.home_advantage
        
        # Score attendu
        expected_home = self.calculate_expected_score(home_rating, away_rating, home_adv)
        expected_away = 1 - expected_home
        
        # Multiplicateur marge de victoire
        goal_diff = home_score - away_score
        rating_diff = home_rating - away_rating
        mov_multiplier = self.margin_of_victory_multiplier(goal_diff, rating_diff)
        
        # Mise à jour Elo
        k = self.k_factor * mov_multiplier
        home_elo_change = k * (actual_home - expected_home)
        away_elo_change = k * (actual_away - expected_away)
        
        self.team_ratings[home_id]['elo'] += home_elo_change
        self.team_ratings[away_id]['elo'] -= home_elo_change  # Conservation somme
        
        # Mettre à jour les compteurs
        self.team_ratings[home_id]['games'] += 1
        self.team_ratings[away_id]['games'] += 1
        self.team_ratings[home_id]['last_update'] = match_date
        self.team_ratings[away_id]['last_update'] = match_date
        
        # Enregistrer l'historique
        match_record = {
            'date': match_date,
            'home_id': home_id,
            'away_id': away_id,
            'home_score': home_score,
            'away_score': away_score,
            'home_elo_before': home_rating,
            'away_elo_before': away_rating,
            'home_elo_change': home_elo_change,
            'home_elo_after': self.team_ratings[home_id]['elo'],
            'away_elo_after': self.team_ratings[away_id]['elo']
        }
        
        if home_id not in self.team_history:
            self.team_history[home_id] = []
        if away_id not in self.team_history:
            self.team_history[away_id] = []
        
        self.team_history[home_id].append(match_record)
        self.team_history[away_id].append(match_record)
        
        return home_elo_change
    
    def apply_time_decay(self, team_id: int, current_date: datetime):
        """Applique un décay temporel au rating"""
        if team_id in self.team_ratings:
            last_update = self.team_ratings[team_id]['last_update']
            if last_update:
                days_since = (current_date - last_update).days
                if days_since > 30:  # Décay après 30 jours sans match
                    decay_factor = 0.99 ** (days_since / 30)
                    current_elo = self.team_ratings[team_id]['elo']
                    base_elo = 1500
                    self.team_ratings[team_id]['elo'] = current_elo * decay_factor + base_elo * (1 - decay_factor)
    
    def get_team_form(self, team_id: int, last_n: int = 5) -> float:
        """Calcule la forme sur les N derniers matchs"""
        if team_id not in self.team_history or len(self.team_history[team_id]) < last_n:
            return 0.5
        
        recent_matches = self.team_history[team_id][-last_n:]
        points = 0
        
        for match in recent_matches:
            is_home = match['home_id'] == team_id
            
            if is_home:
                home_score = match['home_score']
                away_score = match['away_score']
            else:
                home_score = match['away_score']
                away_score = match['home_score']
            
            if home_score > away_score:
                points += 3
            elif home_score == away_score:
                points += 1
        
        max_points = last_n * 3
        return points / max_points if max_points > 0 else 0.5
    
    def predict_match(self, home_id: int, away_id: int, 
                     home_form: float = None, away_form: float = None,
                     is_neutral: bool = False) -> Dict:
        """Prédit les probabilités d'un match"""
        
        # Ratings de base
        home_elo = self.team_ratings.get(home_id, {}).get('elo', 1500)
        away_elo = self.team_ratings.get(away_id, {}).get('elo', 1500)
        
        # Ajustement par la forme
        if home_form is None:
            home_form = self.get_team_form(home_id, 5)
        if away_form is None:
            away_form = self.get_team_form(away_id, 5)
        
        form_adjustment = (home_form - 0.5) * 50 - (away_form - 0.5) * 50
        
        # Avantage terrain
        home_adv = 0 if is_neutral else self.home_advantage
        
        # Rating effectif avec forme
        effective_home = home_elo + form_adjustment + home_adv
        effective_away = away_elo - form_adjustment
        
        # Probabilité victoire domicile
        home_win_prob = self.calculate_expected_score(effective_home, effective_away)
        
        # Probabilité match nul (basée sur différence Elo)
        elo_diff = abs(effective_home - effective_away)
        draw_prob = 0.25 * np.exp(-elo_diff / 200)
        
        # Normalisation
        total = home_win_prob + draw_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob = 1 - home_win_prob - draw_prob
        
        # Confiance basée sur la différence Elo
        confidence = 1 - 1 / (1 + np.exp(-abs(elo_diff) / 100))
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'home_form': home_form,
            'away_form': away_form,
            'elo_diff': elo_diff,
            'confidence': confidence
        }
    
    def train_from_fixtures(self, fixtures: List[Dict]):
        """Entraîne le système Elo sur des matchs historiques"""
        for fixture in fixtures:
            home_id = fixture.get('home_id')
            away_id = fixture.get('away_id')
            home_score = fixture.get('home_score')
            away_score = fixture.get('away_score')
            date_str = fixture.get('date')
            
            if all([home_id, away_id, home_score is not None, away_score is not None, date_str]):
                try:
                    match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    self.update_ratings(home_id, away_id, home_score, away_score, match_date)
                except:
                    continue

# =============================================================================
# VALUE BET DETECTOR
# =============================================================================

class ValueBetDetector:
    """Détecteur de value bets"""
    
    def __init__(self, elo_system: AdvancedEloSystem):
        self.elo = elo_system
        self.min_edge = 0.02  # 2% minimum
        self.min_confidence = 0.60  # 60% minimum
        
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
            
            # Analyser Over/Under si disponible
            ou_analysis = self._analyze_over_under(prediction, bookmaker_odds)
            if ou_analysis:
                value_bets.extend(ou_analysis)
            
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
            
            # Prendre le premier bookmaker (généralement le plus fiable)
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
                
                elif 'over' in market or 'under' in market:
                    # Extraire le nombre de buts
                    import re
                    numbers = re.findall(r'\d+\.?\d*', market)
                    if numbers:
                        threshold = float(numbers[0])
                        for value in bet.get('values', []):
                            outcome = value.get('value', '').lower()
                            odd = float(value.get('odd', 1))
                            
                            if 'over' in outcome:
                                bets[f'over_{threshold}'] = odd
                            elif 'under' in outcome:
                                bets[f'under_{threshold}'] = odd
            
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
                
                if edge >= self.min_edge and prediction['confidence'] >= self.min_confidence:
                    value_bet = {
                        'market': '1X2',
                        'selection': market['name'],
                        'probability': market['prob'],
                        'odds': odds_value,
                        'edge': edge,
                        'implied_prob': 1 / odds_value,
                        'value_score': edge * market['prob'] * 100,
                        'confidence': prediction['confidence'],
                        'expected_value': (market['prob'] * odds_value) - 1
                    }
                    
                    value_bets.append(value_bet)
        
        return value_bets
    
    def _analyze_over_under(self, prediction: Dict, odds: Dict) -> List[Dict]:
        """Analyse les marchés Over/Under"""
        value_bets = []
        
        # Estimation du nombre de buts attendus
        # Basé sur les probabilités et historique
        expected_goals = 2.5  # Valeur par défaut
        
        # Analyser différents seuils
        thresholds = [1.5, 2.5, 3.5]
        
        for threshold in thresholds:
            over_key = f'over_{threshold}'
            under_key = f'under_{threshold}'
            
            # Probabilité estimée (simplifiée)
            # Dans une vraie implémentation, utiliser un modèle de prédiction de buts
            if expected_goals > threshold:
                over_prob = 0.6  # Estimation
                under_prob = 0.4
            else:
                over_prob = 0.4
                under_prob = 0.6
            
            # Analyser Over
            over_odds = odds.get(over_key)
            if over_odds and over_odds > 1:
                edge = self._calculate_edge(over_prob, over_odds)
                if edge >= self.min_edge:
                    value_bets.append({
                        'market': f'Over {threshold}',
                        'selection': f'O{threshold}',
                        'probability': over_prob,
                        'odds': over_odds,
                        'edge': edge,
                        'value_score': edge * over_prob * 100
                    })
            
            # Analyser Under
            under_odds = odds.get(under_key)
            if under_odds and under_odds > 1:
                edge = self._calculate_edge(under_prob, under_odds)
                if edge >= self.min_edge:
                    value_bets.append({
                        'market': f'Under {threshold}',
                        'selection': f'U{threshold}',
                        'probability': under_prob,
                        'odds': under_odds,
                        'edge': edge,
                        'value_score': edge * under_prob * 100
                    })
        
        return value_bets
    
    def _calculate_edge(self, probability: float, odds: float) -> float:
        """Calcule l'edge (valeur attendue)"""
        if odds <= 1 or probability <= 0:
            return -1
        return (probability * odds) - 1
    
    def calculate_kelly_stake(self, edge: float, odds: float, 
                             bankroll: float, fraction: float = 0.25) -> float:
        """Calcule la mise selon Kelly fractionnaire"""
        if edge <= 0 or odds <= 1:
            return 0.0
        
        b = odds - 1
        p = (edge / b) + (1 / odds)
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Limiter
        
        return kelly_fraction * fraction * bankroll

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
            'win_rate': 0.0,
            'avg_odds': 0.0,
            'avg_edge': 0.0
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
            self.performance['win_rate'] = (self.performance['won'] / 
                                          (self.performance['won'] + self.performance['lost'])) * 100
            
            if self.performance['total_staked'] > 0:
                self.performance['roi'] = (self.performance['total_profit'] / 
                                         self.performance['total_staked']) * 100
        
        # Calcul moyenne des cotes et edge
        settled_bets = [b for b in self.bets if b['status'] == 'settled']
        if settled_bets:
            self.performance['avg_odds'] = np.mean([b['odds'] for b in settled_bets])
            self.performance['avg_edge'] = np.mean([b.get('edge', 0) for b in settled_bets])

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
        font-size: 2.8rem;
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
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .live-badge {
        background: linear-gradient(90deg, #FF416C, #FF4B2B);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
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
        padding: 10px 24px;
        border-radius: 8px;
    }
    .profit-positive { color: #4CAF50; font-weight: bold; }
    .profit-negative { color: #f44336; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">💰 SYSTÈME DE PARIS FOOTBALL PRO</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Votre clé API intégrée • Value Bets en Direct • Élo Rating Avancé</div>', unsafe_allow_html=True)

def main():
    """Application principale"""
    setup_interface()
    
    # Initialisation des composants
    if 'api_client' not in st.session_state:
        st.session_state.api_client = FootballDataClient()
    
    if 'elo_system' not in st.session_state:
        st.session_state.elo_system = AdvancedEloSystem()
    
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
        
        min_confidence = st.slider("Confiance minimum (%)", 50, 95, 65, 5)
        st.session_state.value_detector.min_confidence = min_confidence / 100
        
        kelly_fraction = st.slider("Fraction de Kelly", 0.1, 1.0, 0.25, 0.05)
        
        # Info
        st.divider()
        st.info(f"""
        **📊 VOTRE SYSTÈME**
        - Clé API: ✅ Active
        - Mode: Professionnel
        - Dernière MAJ: {datetime.now().strftime('%H:%M:%S')}
        """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard", 
        "🎯 Value Bets", 
        "⚽ Matchs en Direct", 
        "💰 Paris", 
        "📈 Performances"
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
        # Nombre de matchs aujourd'hui
        st.metric("📅 MATCHS AJD", "N/A")
    
    with col4:
        # Edge moyen détecté
        st.metric("⚡ EDGE MOYEN", "N/A")
    
    # Boutons d'action
    st.subheader("🚀 ACTIONS RAPIDES")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Scanner les value bets", type="primary"):
            st.session_state.scan_value_bets = True
            st.rerun()
    
    with col2:
        if st.button("🔄 Rafraîchir les données"):
            st.session_state.api_client.cache.clear()
            st.rerun()
    
    with col3:
        if st.button("📊 Entraîner le modèle Élo"):
            with st.spinner("Entraînement en cours..."):
                # Charger des matchs historiques
                leagues = st.session_state.api_client.get_leagues()
                if leagues:
                    premier_league = next((l for l in leagues if "Premier League" in l['name']), None)
                    if premier_league:
                        fixtures = st.session_state.api_client.get_fixtures(
                            premier_league['id'],
                            premier_league['season'],
                            last=50
                        )
                        st.session_state.elo_system.train_from_fixtures(fixtures)
                        st.success(f"Modèle entraîné sur {len(fixtures)} matchs !")
    
    # Graphique de performance
    st.subheader("📈 ÉVOLUTION BANKROLL")
    
    if st.session_state.bet_manager and len(st.session_state.bet_manager.bets) > 0:
        # Créer un graphique d'évolution
        bet_history = sorted(st.session_state.bet_manager.bets, key=lambda x: x['timestamp'])
        
        bankroll_history = [st.session_state.bet_manager.initial_bankroll]
        current = st.session_state.bet_manager.initial_bankroll
        dates = [datetime.now().replace(hour=0, minute=0, second=0)]
        
        for bet in bet_history:
            if bet['status'] == 'settled':
                if bet['result'] == 'win':
                    current += bet['potential_win']
                elif bet['result'] == 'loss':
                    current -= bet['stake']
                bankroll_history.append(current)
                dates.append(bet['timestamp'])
        
        if len(bankroll_history) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=bankroll_history,
                mode='lines+markers',
                name='Bankroll',
                line=dict(color='#1E88E5', width=3)
            ))
            
            fig.update_layout(
                title='Évolution du Bankroll',
                xaxis_title='Date',
                yaxis_title='Bankroll (€)',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun pari enregistré. Les graphiques apparaîtront ici.")

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
    
    league_options = {f"{l['name']} ({l['country']})": l for l in filtered_leagues}
    
    selected_league_name = st.selectbox(
        "Choisir une ligue",
        list(league_options.keys()),
        index=0 if league_options else None
    )
    
    if selected_league_name:
        selected_league = league_options[selected_league_name]
        
        # Période
        col1, col2 = st.columns(2)
        with col1:
            days_ahead = st.slider("Jours à venir", 1, 14, 3)
        with col2:
            max_matches = st.slider("Max matchs", 10, 100, 30)
        
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
                        
                        for analysis in sorted(value_bets_found, 
                                             key=lambda x: max([b['value_score'] for b in x['value_bets']], default=0), 
                                             reverse=True):
                            
                            with st.expander(f"🎯 {analysis['match']} - {len(analysis['value_bets'])} opportunités"):
                                display_match_analysis(analysis)
                    
                    else:
                        st.info("Aucun value bet détecté avec les paramètres actuels.")
                        
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {str(e)}")

def display_match_analysis(analysis: Dict):
    """Affiche l'analyse détaillée d'un match"""
    st.write(f"**Ligue:** {analysis['league']}")
    st.write(f"**Date:** {analysis['date'][:10]} {analysis['date'][11:16]}")
    
    # Prédiction
    pred = analysis['prediction']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("1", f"{pred['home_win']*100:.1f}%", 
                 f"Elo: {pred['home_elo']:.0f}")
    
    with col2:
        st.metric("N", f"{pred['draw']*100:.1f}%")
    
    with col3:
        st.metric("2", f"{pred['away_win']*100:.1f}%",
                 f"Elo: {pred['away_elo']:.0f}")
    
    st.write(f"**Confiance:** {pred['confidence']*100:.1f}%")
    
    # Value bets
    st.subheader("💰 Value Bets détectés")
    
    for value_bet in sorted(analysis['value_bets'], key=lambda x: x['value_score'], reverse=True):
        st.markdown(f"""
        <div class="value-bet-card">
            <h4>🎯 {value_bet['market']} - {value_bet['selection']}</h4>
            <p><strong>Edge:</strong> {value_bet['edge']*100:.2f}% • 
            <strong>Cote:</strong> {value_bet['odds']:.2f} • 
            <strong>Score:</strong> {value_bet['value_score']:.1f}</p>
            <p>Probabilité modèle: {value_bet['probability']*100:.1f}% • 
            Probabilité implicite: {value_bet['implied_prob']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calcul de la mise Kelly
        if st.session_state.bet_manager:
            bankroll = st.session_state.bet_manager.bankroll
            kelly_stake = st.session_state.value_detector.calculate_kelly_stake(
                value_bet['edge'],
                value_bet['odds'],
                bankroll,
                fraction=0.25
            )
            
            if kelly_stake > 0:
                st.info(f"**Mise Kelly recommandée:** €{kelly_stake:,.2f} ({(kelly_stake/bankroll*100):.1f}% du bankroll)")
                
                # Bouton pour placer le pari
                if st.button(f"📝 Parier {value_bet['selection']} @ {value_bet['odds']:.2f}", 
                           key=f"bet_{analysis['fixture_id']}_{value_bet['market']}"):
                    
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
                    
                    result = st.session_state.bet_manager.place_bet(
                        match_info, bet_details, kelly_stake, value_bet['odds']
                    )
                    
                    if result['success']:
                        st.success(f"✅ Pari placé ! Mise: €{kelly_stake:,.2f}")
                    else:
                        st.error(f"❌ Erreur: {result.get('error')}")

def display_live_matches():
    """Affiche les matchs en direct"""
    st.header("⚽ MATCHS EN DIRECT")
    
    if st.button("🔄 Actualiser les matchs en direct"):
        st.session_state.api_client.cache.clear()
    
    with st.spinner("Chargement des matchs en direct..."):
        live_matches = st.session_state.api_client.get_live_fixtures()
        
        if not live_matches:
            st.info("Aucun match en cours actuellement.")
            return
        
        st.success(f"📡 {len(live_matches)} match(s) en direct")
        
        for match in live_matches:
            with st.expander(f"🔥 {match['home_name']} {match['home_score']} - {match['away_score']} {match['away_name']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Statut:** {match['status'].get('long', 'En cours')}")
                    st.write(f"**Temps:** {match['elapsed']}'")
                    
                    # Prédiction
                    prediction = st.session_state.elo_system.predict_match(
                        match['home_id'], match['away_id']
                    )
                    
                    st.write("**Prédiction:**")
                    st.write(f"1: {prediction['home_win']*100:.1f}%")
                    st.write(f"N: {prediction['draw']*100:.1f}%")
                    st.write(f"2: {prediction['away_win']*100:.1f}%")
                
                with col2:
                    # Événements
                    events = st.session_state.api_client.get_fixture_events(match['fixture_id'])
                    if events:
                        st.write("**Événements récents:**")
                        for event in events[-5:]:  # 5 derniers événements
                            time = event.get('time', {}).get('elapsed', '')
                            type_ = event.get('type', '')
                            team = event.get('team', {}).get('name', '')
                            player = event.get('player', {}).get('name', '')
                            
                            if type_ == 'Goal':
                                st.write(f"⚽ {time}' - {player} ({team})")
                            elif type_ == 'Card':
                                card = event.get('detail', '')
                                st.write(f"🟨 {time}' - {player} ({team}) - {card}")
                
                with col3:
                    # Statistiques
                    stats = st.session_state.api_client.get_fixture_statistics(match['fixture_id'])
                    if stats:
                        st.write("**Statistiques:**")
                        for team_stats in stats:
                            team_name = team_stats.get('team', {}).get('name', '')
                            st.write(f"**{team_name}:**")
                            
                            for stat in team_stats.get('statistics', [])[:3]:
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
        
        stake_method = st.selectbox("Méthode de mise", 
                                   ["Montant fixe", "% du bankroll", "Kelly personnalisé"])
        
        if stake_method == "Montant fixe":
            stake = st.number_input("Mise (€)", min_value=1.0, 
                                   max_value=st.session_state.bet_manager.bankroll, 
                                   value=100.0, step=10.0)
        elif stake_method == "% du bankroll":
            percent = st.slider("Pourcentage", 0.5, 10.0, 2.0, 0.5)
            stake = st.session_state.bet_manager.bankroll * (percent / 100)
            st.write(f"**Mise:** €{stake:,.2f} ({percent}%)")
        else:
            edge = st.number_input("Edge estimé (%)", min_value=0.1, max_value=50.0, 
                                  value=5.0, step=0.5) / 100
            kelly_fraction = st.slider("Fraction de Kelly", 0.1, 1.0, 0.25, 0.05)
            stake = st.session_state.value_detector.calculate_kelly_stake(
                edge, odds, st.session_state.bet_manager.bankroll, kelly_fraction
            )
            st.write(f"**Mise Kelly:** €{stake:,.2f}")
    
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
            'probability': 0.5,  # À estimer
            'edge': edge if stake_method == "Kelly personnalisé" else 0
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
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{bet['match']}**")
                    st.write(f"{bet['market']} - {bet['selection']} @ {bet['odds']:.2f}")
                with col2:
                    st.write(f"**Mise:** €{bet['stake']:,.2f}")
                with col3:
                    # Options pour régler le pari
                    result_col1, result_col2, result_col3 = st.columns(3)
                    with result_col1:
                        if st.button("✅", key=f"win_{bet['id']}"):
                            st.session_state.bet_manager.settle_bet(bet['id'], 'win')
                            st.rerun()
                    with result_col2:
                        if st.button("❌", key=f"loss_{bet['id']}"):
                            st.session_state.bet_manager.settle_bet(bet['id'], 'loss')
                            st.rerun()
                    with result_col3:
                        if st.button("↩️", key=f"void_{bet['id']}"):
                            st.session_state.bet_manager.settle_bet(bet['id'], 'void')
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
    
    # Graphiques
    st.subheader("📊 Graphiques de Performance")
    
    if len(st.session_state.bet_manager.bets) > 0:
        # Évolution du bankroll
        bet_history = sorted([b for b in st.session_state.bet_manager.bets if b['status'] == 'settled'], 
                           key=lambda x: x['timestamp'])
        
        if bet_history:
            bankroll_history = [st.session_state.bet_manager.initial_bankroll]
            current = st.session_state.bet_manager.initial_bankroll
            dates = [bet_history[0]['timestamp'].replace(hour=0, minute=0, second=0)]
            
            for bet in bet_history:
                if bet['result'] == 'win':
                    current += bet['potential_win']
                elif bet['result'] == 'loss':
                    current -= bet['stake']
                bankroll_history.append(current)
                dates.append(bet['timestamp'])
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=dates, y=bankroll_history,
                mode='lines+markers',
                name='Bankroll',
                line=dict(color='#1E88E5', width=3)
            ))
            
            fig1.update_layout(
                title='Évolution du Bankroll',
                xaxis_title='Date',
                yaxis_title='Bankroll (€)',
                template='plotly_white'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Distribution des gains
            profits = []
            for bet in bet_history:
                if bet['result'] == 'win':
                    profits.append(bet['potential_win'] - bet['stake'])
                else:
                    profits.append(-bet['stake'])
            
            if profits:
                fig2 = px.histogram(
                    x=profits,
                    nbins=20,
                    title='Distribution des Profits/Perte par Pari',
                    labels={'x': 'Profit (€)', 'y': 'Nombre de Paris'}
                )
                
                fig2.update_layout(
                    template='plotly_white',
                    showlegend=False
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    # Détails des paris
    st.subheader("📋 Historique détaillé")
    
    if st.session_state.bet_manager.bets:
        history_data = []
        
        for bet in st.session_state.bet_manager.bets:
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
                'Profit': f"€{bet.get('potential_win', 0):,.2f}" if bet.get('result') == 'win' else f"-€{bet['stake']:,.2f}"
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Bouton d'export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter en CSV",
            data=csv,
            file_name=f"historique_paris_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucun pari dans l'historique.")

# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == "__main__":
    main()
