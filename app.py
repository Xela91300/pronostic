# app.py - Système de Paris Sportifs en Temps Réel
# Version Professionnelle avec API réelles

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
import os
from dataclasses import dataclass
import threading
from queue import Queue
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# CONFIGURATION DES API
# =============================================================================

@dataclass
class APIConfig:
    """Configuration des APIs de bookmakers"""
    
    # API-Football (api-sports.io)
    API_FOOTBALL_KEY: str = st.secrets.get("API_FOOTBALL_KEY", "")
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"
    
    # The Odds API (pour les cotes en direct)
    ODDS_API_KEY: str = st.secrets.get("ODDS_API_KEY", "")
    ODDS_API_URL: str = "https://api.the-odds-api.com/v4"
    
    # Betfair Exchange API (pour l'échange)
    BETFAIR_KEY: str = st.secrets.get("BETFAIR_KEY", "")
    
    # RapidAPI alternatives
    RAPIDAPI_KEY: str = st.secrets.get("RAPIDAPI_KEY", "")
    
    # Cache configuration
    CACHE_TTL: int = 60  # 1 minute pour les données en direct
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 10

# =============================================================================
# CLIENT API PROFESSIONNEL
# =============================================================================

class ProfessionalOddsClient:
    """Client professionnel pour récupérer les cotes en temps réel"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache = {}
        self.cache_timestamps = {}
    
    def get_live_odds(self, sport: str = 'soccer', regions: str = 'eu', 
                      markets: str = 'h2h,spreads,totals') -> List[Dict]:
        """Récupère les cotes en direct de The Odds API"""
        
        cache_key = f"odds_{sport}_{regions}_{markets}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.ODDS_API_URL}/sports/{sport}/odds"
            params = {
                'apiKey': self.config.ODDS_API_KEY,
                'regions': regions,
                'markets': markets,
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            self._cache_data(cache_key, data)
            
            return self._process_odds_data(data)
            
        except Exception as e:
            st.error(f"Erreur API Odds: {str(e)}")
            return []
    
    def get_fixtures(self, league_id: int, season: int, 
                    from_date: str, to_date: str) -> List[Dict]:
        """Récupère les matchs de l'API-Football"""
        
        cache_key = f"fixtures_{league_id}_{season}_{from_date}_{to_date}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            headers = {
                'x-apisports-key': self.config.API_FOOTBALL_KEY
            }
            params = {
                'league': league_id,
                'season': season,
                'from': from_date,
                'to': to_date,
                'timezone': 'Europe/Paris'
            }
            
            response = self.session.get(url, headers=headers, params=params, 
                                       timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json().get('response', [])
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            st.error(f"Erreur API Fixtures: {str(e)}")
            return []
    
    def get_live_fixtures(self) -> List[Dict]:
        """Récupère les matchs en cours"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            headers = {
                'x-apisports-key': self.config.API_FOOTBALL_KEY
            }
            params = {
                'live': 'all',
                'timezone': 'Europe/Paris'
            }
            
            response = self.session.get(url, headers=headers, params=params,
                                       timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            return response.json().get('response', [])
            
        except Exception as e:
            st.error(f"Erreur matchs en direct: {str(e)}")
            return []
    
    def get_fixture_statistics(self, fixture_id: int) -> Dict:
        """Récupère les statistiques d'un match"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures/statistics"
            headers = {
                'x-apisports-key': self.config.API_FOOTBALL_KEY
            }
            params = {'fixture': fixture_id}
            
            response = self.session.get(url, headers=headers, params=params,
                                       timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            return response.json().get('response', {})
            
        except Exception as e:
            st.warning(f"Statistiques non disponibles: {str(e)}")
            return {}
    
    def get_fixture_events(self, fixture_id: int) -> List[Dict]:
        """Récupère les événements d'un match (buts, cartons)"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures/events"
            headers = {
                'x-apisports-key': self.config.API_FOOTBALL_KEY
            }
            params = {'fixture': fixture_id}
            
            response = self.session.get(url, headers=headers, params=params,
                                       timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            return response.json().get('response', [])
            
        except Exception as e:
            st.warning(f"Événements non disponibles: {str(e)}")
            return []
    
    def _process_odds_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Traite les données de cotes"""
        processed = []
        
        for match in raw_data:
            try:
                # Récupérer les meilleures cotes par marché
                best_odds = self._extract_best_odds(match.get('bookmakers', []))
                
                processed_match = {
                    'id': match.get('id'),
                    'sport_key': match.get('sport_key'),
                    'commence_time': match.get('commence_time'),
                    'home_team': match.get('home_team'),
                    'away_team': match.get('away_team'),
                    'best_odds': best_odds,
                    'bookmakers': [b.get('key') for b in match.get('bookmakers', [])],
                    'last_updated': datetime.now().isoformat()
                }
                
                processed.append(processed_match)
                
            except Exception as e:
                continue
        
        return processed
    
    def _extract_best_odds(self, bookmakers: List[Dict]) -> Dict:
        """Extrait les meilleures cotes parmi tous les bookmakers"""
        best_odds = {
            'home': {'odds': 0, 'bookmaker': None},
            'draw': {'odds': 0, 'bookmaker': None},
            'away': {'odds': 0, 'bookmaker': None}
        }
        
        for bookmaker in bookmakers:
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    for outcome in market.get('outcomes', []):
                        outcome_name = outcome.get('name', '').lower()
                        odds = outcome.get('price', 0)
                        
                        if 'home' in outcome_name and odds > best_odds['home']['odds']:
                            best_odds['home'] = {'odds': odds, 'bookmaker': bookmaker.get('key')}
                        elif 'draw' in outcome_name and odds > best_odds['draw']['odds']:
                            best_odds['draw'] = {'odds': odds, 'bookmaker': bookmaker.get('key')}
                        elif 'away' in outcome_name and odds > best_odds['away']['odds']:
                            best_odds['away'] = {'odds': odds, 'bookmaker': bookmaker.get('key')}
        
        return best_odds
    
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
# MODÈLE DE PRÉDICTION AVANCÉ
# =============================================================================

class AdvancedPredictionModel:
    """Modèle de prédiction avancé avec machine learning"""
    
    def __init__(self):
        self.models = {}
        self.feature_encoder = {}
        self.scaler = None
        self.team_ratings = {}
        
    def train_model(self, historical_data: pd.DataFrame):
        """Entraîne le modèle sur les données historiques"""
        try:
            # Préparation des features
            features, labels = self._prepare_training_data(historical_data)
            
            if len(features) < 100:
                st.warning("Données insuffisantes pour l'entraînement")
                return
            
            # Entraînement du modèle
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # Normalisation
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Modèle Gradient Boosting
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=20,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Évaluation
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            self.models['gradient_boosting'] = model
            
            st.success(f"""
            🎯 Modèle entraîné avec succès:
            - Accuracy (train): {train_score:.3f}
            - Accuracy (test): {test_score:.3f}
            - Échantillons: {len(features)}
            """)
            
        except Exception as e:
            st.error(f"Erreur entraînement modèle: {str(e)}")
    
    def predict_match(self, home_team: str, away_team: str, 
                     match_context: Dict) -> Dict:
        """Prédit les probabilités d'un match"""
        try:
            # Features pour la prédiction
            features = self._extract_match_features(home_team, away_team, match_context)
            
            if self.scaler and 'gradient_boosting' in self.models:
                features_scaled = self.scaler.transform([features])
                model = self.models['gradient_boosting']
                
                # Probabilités
                probabilities = model.predict_proba(features_scaled)[0]
                
                return {
                    'home_win': float(probabilities[2]),  # Index pour victoire domicile
                    'draw': float(probabilities[1]),      # Index pour match nul
                    'away_win': float(probabilities[0]),  # Index pour victoire extérieur
                    'confidence': float(np.max(probabilities)),
                    'features_used': len(features)
                }
            
            # Fallback: modèle Elo simple
            return self._elo_prediction(home_team, away_team, match_context)
            
        except Exception as e:
            st.error(f"Erreur prédiction: {str(e)}")
            return {'home_win': 0.33, 'draw': 0.34, 'away_win': 0.33, 'confidence': 0.5}
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple:
        """Prépare les données d'entraînement"""
        features_list = []
        labels_list = []
        
        for _, match in data.iterrows():
            try:
                features = self._extract_match_features(
                    match['home_team'],
                    match['away_team'],
                    match.to_dict()
                )
                
                # Label: résultat du match
                if match['home_score'] > match['away_score']:
                    label = 2  # Victoire domicile
                elif match['home_score'] < match['away_score']:
                    label = 0  # Victoire extérieur
                else:
                    label = 1  # Match nul
                
                features_list.append(features)
                labels_list.append(label)
                
            except Exception as e:
                continue
        
        return np.array(features_list), np.array(labels_list)
    
    def _extract_match_features(self, home_team: str, away_team: str, 
                               context: Dict) -> np.ndarray:
        """Extrait les features d'un match"""
        features = []
        
        # 1. Ratings Elo (simplifiés)
        home_elo = self.team_ratings.get(home_team, 1500)
        away_elo = self.team_ratings.get(away_team, 1500)
        features.extend([home_elo, away_elo, home_elo - away_elo])
        
        # 2. Forme récente (derniers 5 matchs)
        home_form = context.get('home_form', 0.5)
        away_form = context.get('away_form', 0.5)
        features.extend([home_form, away_form, home_form - away_form])
        
        # 3. Statistiques offensives/défensives
        home_goals_avg = context.get('home_goals_avg', 1.5)
        away_goals_avg = context.get('away_goals_avg', 1.5)
        home_conceded_avg = context.get('home_conceded_avg', 1.2)
        away_conceded_avg = context.get('away_conceded_avg', 1.2)
        features.extend([
            home_goals_avg - away_conceded_avg,
            away_goals_avg - home_conceded_avg
        ])
        
        # 4. Facteurs contextuels
        is_weekend = 1 if context.get('is_weekend', False) else 0
        is_derby = 1 if context.get('is_derby', False) else 0
        features.extend([is_weekend, is_derby])
        
        return np.array(features)
    
    def _elo_prediction(self, home_team: str, away_team: str, 
                       context: Dict) -> Dict:
        """Prédiction basée sur le système Elo"""
        home_elo = self.team_ratings.get(home_team, 1500)
        away_elo = self.team_ratings.get(away_team, 1500)
        
        # Avantage terrain
        home_advantage = 70
        
        # Probabilité victoire domicile
        home_win_prob = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))
        
        # Modèle de match nul
        elo_diff = abs(home_elo - away_elo)
        draw_prob = 0.25 * np.exp(-elo_diff / 200)
        
        # Normalisation
        total = home_win_prob + draw_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob = 1 - home_win_prob - draw_prob
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'confidence': max(home_win_prob, draw_prob, away_win_prob)
        }

# =============================================================================
# DÉTECTEUR DE VALUE BETS
# =============================================================================

class ValueBetDetector:
    """Détecteur avancé de value bets"""
    
    def __init__(self, prediction_model: AdvancedPredictionModel):
        self.model = prediction_model
        self.min_edge = 0.02  # Edge minimum de 2%
        self.min_confidence = 0.60  # Confiance minimum de 60%
        
    def analyze_match(self, match_data: Dict, odds_data: Dict) -> Optional[Dict]:
        """Analyse un match pour détecter les value bets"""
        try:
            # Prédiction du modèle
            prediction = self.model.predict_match(
                match_data['home_team'],
                match_data['away_team'],
                match_data
            )
            
            # Récupérer les cotes
            best_odds = odds_data.get('best_odds', {})
            
            # Analyser chaque marché
            value_bets = []
            
            # Marché 1X2
            h2h_value = self._analyze_h2h_market(prediction, best_odds)
            if h2h_value:
                value_bets.append(h2h_value)
            
            # Marché Asian Handicap
            ah_value = self._analyze_asian_handicap(prediction, odds_data)
            if ah_value:
                value_bets.append(ah_value)
            
            # Marché Over/Under
            ou_value = self._analyze_over_under(prediction, odds_data)
            if ou_value:
                value_bets.append(ou_value)
            
            # Trouver le meilleur value bet
            if value_bets:
                best_value = max(value_bets, key=lambda x: x['value_score'])
                
                if (best_value['edge'] >= self.min_edge and 
                    best_value['confidence'] >= self.min_confidence):
                    
                    return {
                        'match': f"{match_data['home_team']} vs {match_data['away_team']}",
                        'league': match_data.get('league', 'Unknown'),
                        'date': match_data.get('date'),
                        'prediction': prediction,
                        'value_bet': best_value,
                        'analysis_time': datetime.now().isoformat()
                    }
            
            return None
            
        except Exception as e:
            st.error(f"Erreur analyse match: {str(e)}")
            return None
    
    def _analyze_h2h_market(self, prediction: Dict, odds: Dict) -> Optional[Dict]:
        """Analyse le marché 1X2"""
        markets = [
            {'type': 'home', 'prob': prediction['home_win'], 'odds_key': 'home'},
            {'type': 'draw', 'prob': prediction['draw'], 'odds_key': 'draw'},
            {'type': 'away', 'prob': prediction['away_win'], 'odds_key': 'away'}
        ]
        
        best_value = None
        max_edge = 0
        
        for market in markets:
            odds_value = odds.get(market['odds_key'], {}).get('odds', 0)
            
            if odds_value > 1:  # Éviter division par zéro
                edge = self._calculate_edge(market['prob'], odds_value)
                
                if edge > max_edge and edge > self.min_edge:
                    max_edge = edge
                    best_value = {
                        'market': '1X2',
                        'selection': market['type'].upper(),
                        'probability': market['prob'],
                        'odds': odds_value,
                        'bookmaker': odds.get(market['odds_key'], {}).get('bookmaker'),
                        'edge': edge,
                        'implied_prob': 1 / odds_value,
                        'value_score': edge * market['prob'] * 100,
                        'confidence': prediction['confidence']
                    }
        
        return best_value
    
    def _analyze_asian_handicap(self, prediction: Dict, odds_data: Dict) -> Optional[Dict]:
        """Analyse le marché Asian Handicap"""
        # Implémentation simplifiée
        # Dans une version complète, on analyserait les différents handicaps
        return None
    
    def _analyze_over_under(self, prediction: Dict, odds_data: Dict) -> Optional[Dict]:
        """Analyse le marché Over/Under"""
        # Implémentation simplifiée
        # Dans une version complète, on prédirait le nombre de buts
        return None
    
    def _calculate_edge(self, probability: float, odds: float) -> float:
        """Calcule l'edge (Expected Value)"""
        if odds <= 1:
            return -1
        return (probability * odds) - 1
    
    def calculate_kelly_stake(self, edge: float, odds: float, 
                             bankroll: float, fraction: float = 0.25) -> float:
        """Calcule la mise selon le critère de Kelly fractionnaire"""
        if edge <= 0 or odds <= 1:
            return 0.0
        
        b = odds - 1
        p = (edge / b) + (1 / odds)
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Limiter
        
        return kelly_fraction * fraction * bankroll

# =============================================================================
# GESTIONNAIRE DE PARIS
# =============================================================================

class BetManager:
    """Gestionnaire professionnel de paris"""
    
    def __init__(self, initial_bankroll: float = 10000.0):
        self.bankroll = initial_bankroll
        self.open_bets = []
        self.bet_history = []
        self.performance = {
            'total_bets': 0,
            'won_bets': 0,
            'lost_bets': 0,
            'void_bets': 0,
            'total_staked': 0.0,
            'total_return': 0.0,
            'total_profit': 0.0,
            'roi': 0.0,
            'avg_odds': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
    def place_bet(self, match_info: Dict, bet_details: Dict, 
                  stake: float, odds: float) -> Dict:
        """Place un pari"""
        try:
            if stake > self.bankroll:
                return {
                    'success': False,
                    'error': 'Fonds insuffisants',
                    'available': self.bankroll,
                    'requested': stake
                }
            
            bet_id = len(self.bet_history) + 1
            bet = {
                'id': bet_id,
                'timestamp': datetime.now().isoformat(),
                'match': match_info['match'],
                'league': match_info['league'],
                'market': bet_details['market'],
                'selection': bet_details['selection'],
                'stake': stake,
                'odds': odds,
                'potential_win': stake * (odds - 1),
                'potential_return': stake * odds,
                'status': 'open',
                'edge': bet_details.get('edge', 0),
                'confidence': bet_details.get('confidence', 0),
                'value_score': bet_details.get('value_score', 0)
            }
            
            # Déduire la mise du bankroll
            self.bankroll -= stake
            self.open_bets.append(bet)
            self.bet_history.append(bet)
            
            # Mettre à jour les statistiques
            self.performance['total_bets'] += 1
            self.performance['total_staked'] += stake
            
            return {
                'success': True,
                'bet_id': bet_id,
                'bet': bet,
                'remaining_bankroll': self.bankroll
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def settle_bet(self, bet_id: int, result: str) -> Dict:
        """Règle un pari (win/loss/void)"""
        try:
            bet = next((b for b in self.open_bets if b['id'] == bet_id), None)
            
            if not bet:
                return {'success': False, 'error': 'Pari non trouvé'}
            
            bet['status'] = 'settled'
            bet['result'] = result
            bet['settled_at'] = datetime.now().isoformat()
            
            # Mettre à jour le bankroll
            if result == 'win':
                winnings = bet['stake'] * bet['odds']
                self.bankroll += winnings
                self.performance['won_bets'] += 1
                self.performance['total_return'] += winnings
                self.performance['total_profit'] += (winnings - bet['stake'])
                
            elif result == 'loss':
                self.performance['lost_bets'] += 1
                
            elif result == 'void':
                # Remboursement de la mise
                self.bankroll += bet['stake']
                self.performance['void_bets'] += 1
                self.performance['total_staked'] -= bet['stake']
            
            # Retirer du open bets
            self.open_bets = [b for b in self.open_bets if b['id'] != bet_id]
            
            # Mettre à jour les métriques
            self._update_performance_metrics()
            
            return {
                'success': True,
                'bet': bet,
                'bankroll': self.bankroll
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _update_performance_metrics(self):
        """Met à jour les métriques de performance"""
        if self.performance['total_staked'] > 0:
            self.performance['roi'] = (
                self.performance['total_profit'] / 
                self.performance['total_staked'] * 100
            )
        
        # Calcul du drawdown
        if self.bet_history:
            running_bankroll = 10000.0  # Initial
            peak = running_bankroll
            max_dd = 0.0
            
            for bet in sorted(self.bet_history, key=lambda x: x['timestamp']):
                if bet['status'] == 'settled':
                    if bet['result'] == 'win':
                        running_bankroll += bet['potential_win']
                    elif bet['result'] == 'loss':
                        running_bankroll -= bet['stake']
                
                if running_bankroll > peak:
                    peak = running_bankroll
                
                dd = (peak - running_bankroll) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            
            self.performance['max_drawdown'] = max_dd

# =============================================================================
# INTERFACE STREAMLIT PROFESSIONNELLE
# =============================================================================

class ProfessionalBettingInterface:
    """Interface professionnelle pour les paris en direct"""
    
    def __init__(self):
        self.api_config = APIConfig()
        self.odds_client = ProfessionalOddsClient(self.api_config)
        self.prediction_model = AdvancedPredictionModel()
        self.value_detector = ValueBetDetector(self.prediction_model)
        self.bet_manager = None
        
    def setup_interface(self):
        """Configure l'interface Streamlit"""
        st.set_page_config(
            page_title="Système de Paris Sportifs Pro",
            page_icon="💰",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # CSS professionnel
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            padding: 10px;
        }
        .sub-header {
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
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .value-bet-highlight {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            color: white;
            border-left: 5px solid #4CAF50;
            margin: 10px 0;
        }
        .match-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #e9ecef;
            transition: all 0.3s ease;
        }
        .match-card:hover {
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .odds-bubble {
            display: inline-block;
            background: #1E88E5;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            margin: 0 5px;
            font-weight: bold;
        }
        .profit-positive {
            color: #4CAF50;
            font-weight: bold;
        }
        .profit-negative {
            color: #f44336;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1E88E5, #0D47A1);
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 136, 229, 0.4);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header avec mise à jour en temps réel
        st.markdown('<div class="main-header">💰 SYSTÈME DE PARIS SPORTIFS PROFESSIONNEL</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">🎯 Value Bets en Temps Réel • IA Avancée • Gestion Bankroll Pro</div>', unsafe_allow_html=True)
        
        # Indicateur de connexion API
        col1, col2, col3 = st.columns(3)
        with col2:
            if self.api_config.API_FOOTBALL_KEY and self.api_config.ODDS_API_KEY:
                st.markdown('<div class="live-badge">✅ CONNECTÉ AUX BOOKMAKERS</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Clés API manquantes. Configurez les secrets Streamlit.")
    
    def display_dashboard(self):
        """Affiche le dashboard principal"""
        st.header("📊 DASHBOARD PROFESSIONNEL")
        
        # Métriques en temps réel
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bankroll = self.bet_manager.bankroll if self.bet_manager else 10000
            st.metric(
                "💰 BANKROLL",
                f"€{bankroll:,.2f}",
                delta=None
            )
        
        with col2:
            if self.bet_manager:
                roi = self.bet_manager.performance['roi']
                st.metric("📈 ROI", f"{roi:.2f}%")
            else:
                st.metric("📈 ROI", "0.00%")
        
        with col3:
            # Nombre de value bets détectés aujourd'hui
            st.metric("🎯 VALUE BETS", "N/A")
        
        with col4:
            # Edge moyen
            st.metric("⚡ EDGE MOYEN", "N/A")
        
        # Widget de mise à jour en temps réel
        st.subheader("🔄 MISE À JOUR EN TEMPS RÉEL")
        
        refresh_col1, refresh_col2 = st.columns([3, 1])
        with refresh_col1:
            auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        with refresh_col2:
            if st.button("🔄 Rafraîchir maintenant", type="secondary"):
                st.rerun()
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    def display_live_matches(self):
        """Affiche les matchs en direct"""
        st.header("⚽ MATCHS EN DIRECT")
        
        try:
            live_fixtures = self.odds_client.get_live_fixtures()
            
            if not live_fixtures:
                st.info("Aucun match en cours actuellement.")
                return
            
            for fixture in live_fixtures[:10]:  # Limiter à 10 matchs
                fixture_data = fixture.get('fixture', {})
                teams = fixture.get('teams', {})
                goals = fixture.get('goals', {})
                
                with st.expander(f"🔥 {teams.get('home', {}).get('name')} vs {teams.get('away', {}).get('name')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Score:** {goals.get('home', 0)} - {goals.get('away', 0)}")
                        st.write(f"**Statut:** {fixture_data.get('status', {}).get('long')}")
                        st.write(f"**Temps:** {fixture_data.get('status', {}).get('elapsed')}'")
                    
                    with col2:
                        # Statistiques en direct
                        stats = self.odds_client.get_fixture_statistics(fixture_data.get('id'))
                        if stats:
                            st.write("**Statistiques:**")
                            for team_stats in stats:
                                st.write(f"{team_stats.get('team', {}).get('name')}:")
                                for stat in team_stats.get('statistics', [])[:3]:
                                    st.write(f"  {stat.get('type')}: {stat.get('value')}")
                    
                    with col3:
                        # Cotes en direct
                        odds = self.odds_client.get_live_odds()
                        # Trouver les cotes pour ce match
                        match_odds = self._find_match_odds(odds, 
                                                          teams.get('home', {}).get('name'),
                                                          teams.get('away', {}).get('name'))
                        
                        if match_odds:
                            st.write("**Cotes en direct:**")
                            best = match_odds.get('best_odds', {})
                            st.write(f"1: {best.get('home', {}).get('odds', 'N/A')}")
                            st.write(f"X: {best.get('draw', {}).get('odds', 'N/A')}")
                            st.write(f"2: {best.get('away', {}).get('odds', 'N/A')}")
                        
                        # Bouton pour analyser ce match
                        if st.button("📊 Analyser ce match", key=f"analyze_{fixture_data.get('id')}"):
                            self.analyze_specific_match(fixture)
        
        except Exception as e:
            st.error(f"Erreur chargement matchs en direct: {str(e)}")
    
    def display_upcoming_matches(self):
        """Affiche les matchs à venir"""
        st.header("📅 MATCHS À VENIR")
        
        # Sélection de la ligue
        leagues = {
            "Premier League": 39,
            "La Liga": 140,
            "Serie A": 135,
            "Bundesliga": 78,
            "Ligue 1": 61,
            "Champions League": 2
        }
        
        selected_league = st.selectbox("Sélectionnez une ligue", list(leagues.keys()))
        league_id = leagues[selected_league]
        
        # Période
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input("Du", value=date.today())
        with col2:
            to_date = st.date_input("Au", value=date.today() + timedelta(days=7))
        
        if st.button("🔍 Charger les matchs", type="primary"):
            with st.spinner("Chargement des matchs et cotes..."):
                try:
                    # Récupérer les matchs
                    fixtures = self.odds_client.get_fixtures(
                        league_id, 2024,  # Saison actuelle
                        from_date.strftime('%Y-%m-%d'),
                        to_date.strftime('%Y-%m-%d')
                    )
                    
                    # Récupérer les cotes
                    odds_data = self.odds_client.get_live_odds()
                    
                    # Afficher les matchs
                    for fixture in fixtures[:20]:  # Limiter à 20 matchs
                        self._display_match_with_analysis(fixture, odds_data)
                        
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
    
    def display_value_bets(self):
        """Affiche les value bets détectés"""
        st.header("🎯 VALUE BETS DÉTECTÉS")
        
        # Paramètres de détection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_edge = st.slider("Edge minimum (%)", 1.0, 10.0, 2.0, 0.5)
            self.value_detector.min_edge = min_edge / 100
        
        with col2:
            min_confidence = st.slider("Confiance minimum (%)", 50, 95, 65, 5)
            self.value_detector.min_confidence = min_confidence / 100
        
        with col3:
            bankroll_percent = st.slider("Max % bankroll", 1, 10, 2, 1)
        
        if st.button("🔍 Analyser les matchs disponibles", type="primary"):
            with st.spinner("Analyse en cours..."):
                try:
                    # Récupérer les données
                    fixtures = self.odds_client.get_fixtures(
                        39, 2024,  # Premier League
                        date.today().strftime('%Y-%m-%d'),
                        (date.today() + timedelta(days=3)).strftime('%Y-%m-%d')
                    )
                    
                    odds_data = self.odds_client.get_live_odds()
                    
                    # Analyser chaque match
                    value_bets_found = []
                    
                    for fixture in fixtures:
                        match_data = self._prepare_match_data(fixture)
                        match_odds = self._find_match_odds(odds_data, 
                                                          match_data['home_team'],
                                                          match_data['away_team'])
                        
                        if match_odds:
                            analysis = self.value_detector.analyze_match(match_data, match_odds)
                            
                            if analysis:
                                value_bets_found.append(analysis)
                    
                    # Afficher les résultats
                    if value_bets_found:
                        st.success(f"✅ {len(value_bets_found)} value bets détectés!")
                        
                        for bet in sorted(value_bets_found, 
                                         key=lambda x: x['value_bet']['value_score'], 
                                         reverse=True):
                            self._display_value_bet(bet, bankroll_percent)
                    else:
                        st.info("Aucun value bet détecté avec les paramètres actuels.")
                        
                except Exception as e:
                    st.error(f"Erreur analyse: {str(e)}")
    
    def display_betting_interface(self):
        """Interface de placement de paris"""
        st.header("💰 PLACER UN PARI")
        
        if not self.bet_manager:
            initial_bankroll = st.number_input(
                "💰 Bankroll initial (€)",
                min_value=100.0,
                max_value=1000000.0,
                value=10000.0,
                step=500.0
            )
            
            if st.button("Initialiser le bankroll", type="primary"):
                self.bet_manager = BetManager(initial_bankroll)
                st.success(f"Bankroll initialisé: €{initial_bankroll:,.2f}")
                st.rerun()
            
            return
        
        # Interface de pari
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Pari en cours")
            
            match_selection = st.selectbox(
                "Sélectionner un match",
                ["Manchester City vs Liverpool", "Real Madrid vs Barcelona", 
                 "Bayern Munich vs Dortmund"]
            )
            
            market = st.selectbox(
                "Marché",
                ["1X2", "Double Chance", "Over/Under 2.5", "Both Teams to Score"]
            )
            
            selection = st.selectbox(
                "Sélection",
                ["1", "X", "2", "Over", "Under", "Yes", "No"]
            )
        
        with col2:
            st.subheader("📊 Détails du pari")
            
            odds = st.number_input("Cote", min_value=1.01, max_value=100.0, value=2.0, step=0.01)
            
            stake_options = ["Montant fixe", "% du bankroll", "Kelly"]
            stake_method = st.selectbox("Méthode de mise", stake_options)
            
            if stake_method == "Montant fixe":
                stake = st.number_input("Mise (€)", min_value=1.0, 
                                       max_value=self.bet_manager.bankroll, 
                                       value=100.0, step=10.0)
            elif stake_method == "% du bankroll":
                percent = st.slider("Pourcentage", 0.5, 10.0, 2.0, 0.5)
                stake = self.bet_manager.bankroll * (percent / 100)
                st.write(f"Mise: €{stake:,.2f} ({percent}%)")
            else:  # Kelly
                edge = st.number_input("Edge estimé (%)", min_value=0.1, max_value=50.0, 
                                      value=5.0, step=0.5) / 100
                kelly_fraction = st.slider("Fraction de Kelly", 0.1, 1.0, 0.25, 0.05)
                stake = self.value_detector.calculate_kelly_stake(
                    edge, odds, self.bet_manager.bankroll, kelly_fraction
                )
                st.write(f"Mise Kelly: €{stake:,.2f}")
            
            potential_return = stake * odds
            potential_profit = potential_return - stake
            
            st.metric("Retour potentiel", f"€{potential_return:,.2f}")
            st.metric("Profit potentiel", f"€{potential_profit:,.2f}")
            
            if st.button("✅ Placer le pari", type="primary"):
                bet_details = {
                    'market': market,
                    'selection': selection,
                    'edge': edge if stake_method == "Kelly" else 0
                }
                
                match_info = {
                    'match': match_selection,
                    'league': 'Premier League'
                }
                
                result = self.bet_manager.place_bet(match_info, bet_details, stake, odds)
                
                if result['success']:
                    st.success(f"Pari placé! ID: {result['bet_id']}")
                    st.write(f"Bankroll restant: €{result['remaining_bankroll']:,.2f}")
                else:
                    st.error(f"Erreur: {result.get('error', 'Inconnue')}")
    
    def display_performance(self):
        """Affiche les performances"""
        st.header("📈 PERFORMANCES")
        
        if not self.bet_manager:
            st.info("Initialisez d'abord le bankroll pour voir les performances.")
            return
        
        perf = self.bet_manager.performance
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total paris", perf['total_bets'])
        
        with col2:
            win_rate = (perf['won_bets'] / perf['total_bets'] * 100) if perf['total_bets'] > 0 else 0
            st.metric("Taux de réussite", f"{win_rate:.1f}%")
        
        with col3:
            st.metric("ROI total", f"{perf['roi']:.2f}%")
        
        with col4:
            st.metric("Profit total", f"€{perf['total_profit']:,.2f}")
        
        # Graphique d'évolution du bankroll
        st.subheader("📊 Évolution du bankroll")
        
        if self.bet_manager.bet_history:
            # Simuler l'évolution
            bankroll_history = [10000.0]  # Initial
            current = 10000.0
            
            for bet in sorted(self.bet_manager.bet_history, key=lambda x: x['timestamp']):
                if bet['status'] == 'settled':
                    if bet['result'] == 'win':
                        current += bet['potential_win']
                    elif bet['result'] == 'loss':
                        current -= bet['stake']
                    elif bet['result'] == 'void':
                        current += bet['stake']
                
                bankroll_history.append(current)
            
            # Créer le graphique
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(bankroll_history))),
                y=bankroll_history,
                mode='lines+markers',
                name='Bankroll',
                line=dict(color='#1E88E5', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title='Évolution du Bankroll',
                xaxis_title='Nombre de paris',
                yaxis_title='Bankroll (€)',
                template='plotly_white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Détails des paris
        st.subheader("📋 Historique des paris")
        
        if self.bet_manager.bet_history:
            history_df = pd.DataFrame(self.bet_manager.bet_history)
            
            # Formater les colonnes
            display_cols = ['id', 'match', 'market', 'selection', 'stake', 
                          'odds', 'status', 'result', 'edge', 'value_score']
            
            available_cols = [col for col in display_cols if col in history_df.columns]
            
            st.dataframe(
                history_df[available_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Aucun pari dans l'historique.")
    
    def _prepare_match_data(self, fixture: Dict) -> Dict:
        """Prépare les données d'un match pour l'analyse"""
        fixture_data = fixture.get('fixture', {})
        teams = fixture.get('teams', {})
        goals = fixture.get('goals', {})
        
        return {
            'fixture_id': fixture_data.get('id'),
            'home_team': teams.get('home', {}).get('name'),
            'away_team': teams.get('away', {}).get('name'),
            'date': fixture_data.get('date'),
            'status': fixture_data.get('status', {}).get('short'),
            'home_score': goals.get('home'),
            'away_score': goals.get('away'),
            'league': fixture.get('league', {}).get('name'),
            'home_form': 0.5,  # À implémenter avec données historiques
            'away_form': 0.5,
            'home_goals_avg': 1.5,  # À implémenter
            'away_goals_avg': 1.5,
            'home_conceded_avg': 1.2,
            'away_conceded_avg': 1.2,
            'is_weekend': datetime.fromisoformat(fixture_data.get('date').replace('Z', '+00:00')).weekday() >= 5,
            'is_derby': False  # À implémenter
        }
    
    def _find_match_odds(self, odds_list: List[Dict], home_team: str, away_team: str) -> Optional[Dict]:
        """Trouve les cotes d'un match spécifique"""
        for odds in odds_list:
            if (home_team.lower() in odds.get('home_team', '').lower() and 
                away_team.lower() in odds.get('away_team', '').lower()):
                return odds
        return None
    
    def _display_match_with_analysis(self, fixture: Dict, odds_data: List[Dict]):
        """Affiche un match avec analyse"""
        match_data = self._prepare_match_data(fixture)
        
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{match_data['home_team']} vs {match_data['away_team']}**")
                st.write(f"*{match_data['league']} • {match_data['date'][:10]}*")
            
            with col2:
                # Trouver les cotes
                match_odds = self._find_match_odds(odds_data, 
                                                  match_data['home_team'],
                                                  match_data['away_team'])
                
                if match_odds:
                    best = match_odds.get('best_odds', {})
                    st.write("**Meilleures cotes:**")
                    st.write(f"1: {best.get('home', {}).get('odds', 'N/A')}")
                    st.write(f"X: {best.get('draw', {}).get('odds', 'N/A')}")
                    st.write(f"2: {best.get('away', {}).get('odds', 'N/A')}")
            
            with col3:
                if st.button("🎯 Analyser", key=f"analyze_{match_data['fixture_id']}"):
                    if match_odds:
                        analysis = self.value_detector.analyze_match(match_data, match_odds)
                        
                        if analysis:
                            st.success(f"Edge: {analysis['value_bet']['edge']*100:.1f}%")
                            st.write(f"Value Score: {analysis['value_bet']['value_score']:.1f}")
                        else:
                            st.info("Pas de value bet détecté")
            
            st.divider()
    
    def _display_value_bet(self, analysis: Dict, bankroll_percent: float):
        """Affiche un value bet détecté"""
        with st.container():
            st.markdown(f"""
            <div class="value-bet-highlight">
                <h4>🎯 {analysis['match']}</h4>
                <p><strong>Ligue:</strong> {analysis['league']} • 
                <strong>Date:</strong> {analysis['date'][:10]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            value_bet = analysis['value_bet']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📈 Edge", f"{value_bet['edge']*100:.2f}%")
                st.metric("🎯 Value Score", f"{value_bet['value_score']:.1f}")
            
            with col2:
                st.write(f"**Marché:** {value_bet['market']}")
                st.write(f"**Sélection:** {value_bet['selection']}")
                st.write(f"**Bookmaker:** {value_bet.get('bookmaker', 'Multiple')}")
            
            with col3:
                st.write(f"**Probabilité modèle:** {value_bet['probability']*100:.1f}%")
                st.write(f"**Cote:** {value_bet['odds']:.2f}")
                st.write(f"**Probabilité implicite:** {value_bet['implied_prob']*100:.1f}%")
            
            # Calcul de la mise Kelly
            if self.bet_manager:
                kelly_stake = self.value_detector.calculate_kelly_stake(
                    value_bet['edge'],
                    value_bet['odds'],
                    self.bet_manager.bankroll,
                    fraction=0.25
                )
                
                max_stake = self.bet_manager.bankroll * (bankroll_percent / 100)
                recommended_stake = min(kelly_stake, max_stake)
                
                st.info(f"""
                **💰 Mise recommandée:**
                - Kelly (25%): €{kelly_stake:,.2f}
                - Max {bankroll_percent}% bankroll: €{max_stake:,.2f}
                - **Recommandé: €{recommended_stake:,.2f}**
                """)
            
            st.divider()
    
    def analyze_specific_match(self, fixture: Dict):
        """Analyse un match spécifique en détail"""
        st.subheader("📊 Analyse détaillée")
        
        match_data = self._prepare_match_data(fixture)
        odds_data = self.odds_client.get_live_odds()
        match_odds = self._find_match_odds(odds_data, 
                                          match_data['home_team'],
                                          match_data['away_team'])
        
        if match_odds:
            analysis = self.value_detector.analyze_match(match_data, match_odds)
            
            if analysis:
                self._display_value_bet(analysis, 2)
            else:
                st.info("Aucun value bet détecté pour ce match.")
        else:
            st.warning("Cotes non disponibles pour ce match.")

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Fonction principale de l'application"""
    
    # Initialisation
    app = ProfessionalBettingInterface()
    app.setup_interface()
    
    # Vérification des clés API
    if not app.api_config.API_FOOTBALL_KEY or not app.api_config.ODDS_API_KEY:
        st.error("""
        ⚠️ **CLÉS API REQUISES**
        
        Pour utiliser cette application, vous devez configurer:
        
        1. **API-Football** (api-sports.io)
        2. **The Odds API** (the-odds-api.com)
        
        Ajoutez-les dans les secrets Streamlit:
        ```
        API_FOOTBALL_KEY = "votre_clé_api_football"
        ODDS_API_KEY = "votre_clé_odds_api"
        ```
        
        Sans ces clés, l'application fonctionnera en mode démo limité.
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        # Mode de fonctionnement
        mode = st.selectbox(
            "Mode",
            ["🎯 Value Bets", "💰 Paris en direct", "📈 Performance", "⚙️ Configuration API"]
        )
        
        # Paramètres avancés
        with st.expander("🔧 Paramètres avancés"):
            st.checkbox("Analyse approfondie", value=True)
            st.checkbox("Notifications en direct", value=False)
            st.checkbox("Auto-betting (expérimental)", value=False)
        
        # Informations système
        st.divider()
        st.info(f"""
        **📊 Système actif**
        - Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}
        - Mode: {'PRO' if app.api_config.API_FOOTBALL_KEY else 'DÉMO'}
        - Version: 2.1.0
        """)
    
    # Navigation par mode
    if mode == "🎯 Value Bets":
        app.display_value_bets()
        
    elif mode == "💰 Paris en direct":
        tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "⚽ Matchs en direct", "📅 Matchs à venir"])
        
        with tab1:
            app.display_dashboard()
        
        with tab2:
            app.display_live_matches()
        
        with tab3:
            app.display_upcoming_matches()
        
        # Interface de pari
        st.divider()
        app.display_betting_interface()
        
    elif mode == "📈 Performance":
        app.display_performance()
        
    elif mode == "⚙️ Configuration API":
        st.header("Configuration des APIs")
        
        st.info("""
        **Instructions:**
        
        1. **API-Football** (api-sports.io)
           - Inscrivez-vous sur api-sports.io
           - Obtenez votre clé API
           - Limite: 100 requêtes/jour (gratuit)
        
        2. **The Odds API** (the-odds-api.com)
           - Inscrivez-vous sur the-odds-api.com
           - Obtenez votre clé API
           - Limite: 500 requêtes/mois (gratuit)
        
        3. **Configuration Streamlit Secrets:**
           - Allez dans les paramètres de votre app Streamlit
           - Ajoutez les clés dans "Secrets"
        """)
        
        st.code("""
        # .streamlit/secrets.toml
        API_FOOTBALL_KEY = "votre_clé_api_football"
        ODDS_API_KEY = "votre_clé_odds_api"
        """, language="toml")

# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == "__main__":
    main()
