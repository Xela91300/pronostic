# =============================================================================
# PRONOSTIQUEUR MULTI-SPORTS PROFESSIONNEL - Value Bets Advanced
# Version: 3.0 Ultra-Précise - Optimisée pour les paris value
# Fonctionnalités: 
# - Elo/Glicko intégré pour tous les sports
# - Machine Learning avancé avec features engineering complet
# - Gestion de bankroll intelligente (Kelly, etc.)
# - Backtesting intégré
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import time
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import requests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Librairies ML
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb

# =============================================================================
# CONFIGURATION AVANCÉE
# =============================================================================

class Config:
    """Configuration globale ultra-précise"""
    
    # Paramètres Elo/Glicko
    ELO_K_FACTOR = 32  # Facteur d'apprentissage Elo
    ELO_HOME_ADVANTAGE = 70  # Avantage terrain (points Elo)
    ELO_BASE_RATING = 1500
    ELO_REGULARIZATION = 0.3  # Régularisation pour éviter la dérive
    
    # Paramètres ML
    FEATURE_LAG_DAYS = 30  # Fenêtre pour features temporelles
    MIN_TRAINING_SAMPLES = 200  # Minimum pour entraînement
    PROBA_CALIBRATION_SAMPLES = 1000  # Pour calibration des probabilités
    
    # Gestion de bankroll
    KELLY_FRACTION = 0.25  # Fraction de Kelly à utiliser
    MAX_STAKE_PERCENT = 0.02  # Maximum 2% par pari
    MIN_EDGE_THRESHOLD = 0.02  # Edge minimum de 2%
    CONFIDENCE_THRESHOLD = 0.65  # Confiance minimum
    
    # Caches
    CACHE_TTL_BASIC = 1800  # 30 minutes
    CACHE_TTL_ADVANCED = 3600  # 1 heure
    CACHE_TTL_LIVE = 300  # 5 minutes
    
    # API Keys (à configurer dans secrets)
    API_SPORTS_KEY = st.secrets.get("api_sports_key", "")
    ODDS_API_KEY = st.secrets.get("odds_api_key", "")
    
    # Saisons courantes
    NBA_SEASON = "2024-25"
    FOOTBALL_SEASON = 2024
    
# =============================================================================
# SYSTÈMES DE RATING ELO/GLICKO
# =============================================================================

class AdvancedRatingSystem:
    """Système de rating hybride Elo/Glicko avec features avancées"""
    
    def __init__(self, sport: str, decay_factor: float = 0.95):
        self.sport = sport
        self.decay_factor = decay_factor  # Décay des ratings
        self.ratings = {}  # {team_id: {'rating': float, 'rd': float, 'games': int}}
        self.history = {}  # Historique des matchs
        
        # Facteurs sport-spécifiques
        self.sport_factors = {
            'basketball': {'k_factor': 24, 'home_adv': 100, 'margin_factor': 2.2},
            'football': {'k_factor': 32, 'home_adv': 70, 'margin_factor': 1.8},
            'tennis': {'k_factor': 40, 'home_adv': 0, 'margin_factor': 1.5},
        }
        
        self.factors = self.sport_factors.get(sport, self.sport_factors['football'])
    
    def get_initial_rating(self, team_id: int) -> Dict:
        """Initialise un rating pour une nouvelle équipe"""
        return {
            'elo': Config.ELO_BASE_RATING,
            'glicko_rating': Config.ELO_BASE_RATING,
            'glicko_rd': 350,  # Rating Deviation initiale
            'volatility': 0.06,
            'games_played': 0,
            'last_game_date': None,
            'streak': 0,
            'form': 0.5,
            'offensive_rating': 100,
            'defensive_rating': 100
        }
    
    def margin_of_victory_multiplier(self, point_diff: float, rating_diff: float) -> float:
        """Multiplicateur selon la marge de victoire (logistique)"""
        # Formule adaptée de Chess
        return np.log(abs(point_diff) + 1) * (2.2 / (rating_diff * 0.001 + 2.2))
    
    def update_ratings(self, home_id: int, away_id: int, 
                      home_score: int, away_score: int, 
                      game_date: date, venue: str = 'home') -> None:
        """Met à jour les ratings après un match"""
        
        # Initialiser si nécessaire
        if home_id not in self.ratings:
            self.ratings[home_id] = self.get_initial_rating(home_id)
        if away_id not in self.ratings:
            self.ratings[away_id] = self.get_initial_rating(away_id)
        
        home_rating = self.ratings[home_id]
        away_rating = self.ratings[away_id]
        
        # Appliquer décay temporel
        self.apply_time_decay(home_rating, game_date)
        self.apply_time_decay(away_rating, game_date)
        
        # Calculer le résultat
        if home_score > away_score:
            actual_score = 1.0
        elif home_score < away_score:
            actual_score = 0.0
        else:
            actual_score = 0.5  # Match nul
        
        # Ajustement avantage terrain
        home_advantage = self.factors['home_adv'] if venue == 'home' else 0
        
        # Calcul Elo
        expected_home = 1 / (1 + 10 ** ((away_rating['elo'] - home_rating['elo'] - home_advantage) / 400))
        
        # Multiplicateur marge de victoire
        point_diff = home_score - away_score
        mov_multiplier = self.margin_of_victory_multiplier(
            point_diff, 
            home_rating['elo'] - away_rating['elo']
        )
        
        # Mise à jour Elo
        elo_change = self.factors['k_factor'] * mov_multiplier * (actual_score - expected_home)
        home_rating['elo'] += elo_change
        away_rating['elo'] -= elo_change
        
        # Mise à jour form et streak
        self.update_form_and_streak(home_rating, away_rating, actual_score)
        
        # Mise à jour ratings offensifs/défensifs
        self.update_offensive_defensive_ratings(
            home_rating, away_rating, home_score, away_score
        )
        
        # Enregistrer l'historique
        match_key = f"{game_date}_{home_id}_{away_id}"
        self.history[match_key] = {
            'home_elo': home_rating['elo'],
            'away_elo': away_rating['elo'],
            'home_score': home_score,
            'away_score': away_score,
            'result': actual_score
        }
        
        # Mettre à jour le dernier match
        home_rating['last_game_date'] = game_date
        away_rating['last_game_date'] = game_date
        home_rating['games_played'] += 1
        away_rating['games_played'] += 1
    
    def apply_time_decay(self, rating: Dict, current_date: date) -> None:
        """Applique un décay temporel au rating"""
        if rating['last_game_date']:
            days_since = (current_date - rating['last_game_date']).days
            decay = self.decay_factor ** (days_since / 30)  # Décay mensuel
            rating['elo'] = rating['elo'] * decay + Config.ELO_BASE_RATING * (1 - decay)
            rating['form'] *= decay  # Décay de la forme
    
    def update_form_and_streak(self, home_rating: Dict, away_rating: Dict, result: float) -> None:
        """Met à jour la forme et les séries"""
        # Forme sur les 10 derniers matchs (moyenne pondérée)
        home_rating['form'] = home_rating['form'] * 0.9 + result * 0.1
        away_rating['form'] = away_rating['form'] * 0.9 + (1 - result) * 0.1
        
        # Streak
        if result == 1:
            home_rating['streak'] = max(1, home_rating['streak'] + 1)
            away_rating['streak'] = min(-1, away_rating['streak'] - 1)
        elif result == 0:
            home_rating['streak'] = min(-1, home_rating['streak'] - 1)
            away_rating['streak'] = max(1, away_rating['strepe'] + 1)
    
    def update_offensive_defensive_ratings(self, home_rating: Dict, away_rating: Dict,
                                          home_score: int, away_score: int) -> None:
        """Met à jour les ratings offensifs et défensifs"""
        # Moyenne mobile exponentielle
        alpha = 0.15  # Facteur d'apprentissage
        
        home_rating['offensive_rating'] = (home_rating['offensive_rating'] * (1 - alpha) + 
                                          home_score * alpha)
        home_rating['defensive_rating'] = (home_rating['defensive_rating'] * (1 - alpha) + 
                                          away_score * alpha)
        
        away_rating['offensive_rating'] = (away_rating['offensive_rating'] * (1 - alpha) + 
                                          away_score * alpha)
        away_rating['defensive_rating'] = (away_rating['defensive_rating'] * (1 - alpha) + 
                                          home_score * alpha)
    
    def get_match_features(self, home_id: int, away_id: int, venue: str = 'home') -> Dict:
        """Extrait toutes les features de rating pour un match"""
        if home_id not in self.ratings or away_id not in self.ratings:
            return {}
        
        home = self.ratings[home_id]
        away = self.ratings[away_id]
        
        features = {
            'elo_diff': home['elo'] - away['elo'],
            'elo_home': home['elo'],
            'elo_away': away['elo'],
            'form_diff': home['form'] - away['form'],
            'streak_diff': home['streak'] - away['streak'],
            'off_rating_diff': home['offensive_rating'] - away['offensive_rating'],
            'def_rating_diff': home['defensive_rating'] - away['defensive_rating'],
            'net_rating_diff': (home['offensive_rating'] - home['defensive_rating']) - 
                              (away['offensive_rating'] - away['defensive_rating']),
            'games_played_diff': home['games_played'] - away['games_played'],
            'home_advantage': self.factors['home_adv'] if venue == 'home' else 0,
            'momentum': home['form'] * (1 + home['streak'] * 0.1) - 
                       away['form'] * (1 + away['streak'] * 0.1),
        }
        
        return features

# =============================================================================
# COLLECTE DE DONNÉES AVANCÉE
# =============================================================================

class AdvancedDataCollector:
    """Collecteur de données multi-sources avec intégration odds réelles"""
    
    def __init__(self):
        self.sources = {
            'api-football': 'https://v3.football.api-sports.io',
            'odds-api': 'https://api.the-odds-api.com/v4',
            'nba-api': 'https://stats.nba.com/stats'
        }
    
    @st.cache_data(ttl=Config.CACHE_TTL_BASIC)
    def get_football_fixtures(self, league_id: int, season: int, 
                             date_from: str, date_to: str) -> pd.DataFrame:
        """Récupère les matchs de football avec statistiques détaillées"""
        try:
            url = f"{self.sources['api-football']}/fixtures"
            headers = {"x-apisports-key": Config.API_SPORTS_KEY}
            params = {
                'league': league_id,
                'season': season,
                'from': date_from,
                'to': date_to,
                'timezone': 'Europe/Paris'
            }
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            fixtures = []
            for fixture in data.get('response', []):
                # Statistiques détaillées
                stats = self.get_football_statistics(fixture['fixture']['id'])
                
                fixtures.append({
                    'fixture_id': fixture['fixture']['id'],
                    'date': fixture['fixture']['date'],
                    'home_team': fixture['teams']['home']['name'],
                    'away_team': fixture['teams']['away']['name'],
                    'home_id': fixture['teams']['home']['id'],
                    'away_id': fixture['teams']['away']['id'],
                    'league_id': league_id,
                    'season': season,
                    'status': fixture['fixture']['status']['short'],
                    'home_score': fixture['goals']['home'],
                    'away_score': fixture['goals']['away'],
                    **stats
                })
            
            return pd.DataFrame(fixtures)
            
        except Exception as e:
            st.error(f"Erreur API Football: {e}")
            return pd.DataFrame()
    
    def get_football_statistics(self, fixture_id: int) -> Dict:
        """Récupère les statistiques détaillées d'un match"""
        try:
            url = f"{self.sources['api-football']}/fixtures/statistics"
            headers = {"x-apisports-key": Config.API_SPORTS_KEY}
            params = {'fixture': fixture_id}
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            stats = {}
            for team_stats in data.get('response', []):
                team_id = team_stats['team']['id']
                prefix = 'home_' if team_id == 'home' else 'away_'
                
                for stat in team_stats['statistics']:
                    key = prefix + stat['type'].lower().replace(' ', '_')
                    stats[key] = stat['value']
            
            return stats
            
        except:
            return {}
    
    @st.cache_data(ttl=Config.CACHE_TTL_LIVE)
    def get_live_odds(self, sport: str, regions: List[str] = ['eu', 'uk']) -> pd.DataFrame:
        """Récupère les cotes en temps réel"""
        try:
            url = f"{self.sources['odds-api']}/sports/{sport}/odds"
            params = {
                'apiKey': Config.ODDS_API_KEY,
                'regions': ','.join(regions),
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            odds_data = []
            for match in data:
                for bookmaker in match.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'h2h':  # Head to Head
                            for outcome in market['outcomes']:
                                odds_data.append({
                                    'match_id': match['id'],
                                    'sport_key': match['sport_key'],
                                    'home_team': match['home_team'],
                                    'away_team': match['away_team'],
                                    'commence_time': match['commence_time'],
                                    'bookmaker': bookmaker['key'],
                                    'outcome': outcome['name'],
                                    'odds': outcome['price']
                                })
            
            return pd.DataFrame(odds_data)
            
        except Exception as e:
            st.warning(f"Erreur Odds API: {e}")
            return pd.DataFrame()

# =============================================================================
# FEATURE ENGINEERING AVANCÉ
# =============================================================================

class AdvancedFeatureEngineer:
    """Génération de features avancées pour le machine learning"""
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Crée des features temporelles"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Features temporelles
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['hour'] = df[date_col].dt.hour
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Congestion calendaire
        df['days_since_last_game_home'] = df.groupby('home_id')[date_col].diff().dt.days.fillna(7)
        df['days_since_last_game_away'] = df.groupby('away_id')[date_col].diff().dt.days.fillna(7)
        
        return df
    
    @staticmethod
    def create_momentum_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Crée des features de momentum sur une fenêtre glissante"""
        
        features = []
        
        for team_col in ['home_id', 'away_id']:
            for metric in ['goals_for', 'goals_against', 'points']:
                # Moyenne glissante
                df[f'{team_col}_{metric}_ma_{window}'] = df.groupby(team_col)[metric].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                # Écart-type glissante
                df[f'{team_col}_{metric}_std_{window}'] = df.groupby(team_col)[metric].transform(
                    lambda x: x.rolling(window, min_periods=2).std().fillna(0)
                )
                
                # Tendance linéaire
                df[f'{team_col}_{metric}_trend_{window}'] = df.groupby(team_col)[metric].transform(
                    lambda x: x.rolling(window, min_periods=2).apply(
                        lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                    )
                )
        
        return df
    
    @staticmethod
    def create_derived_features(df: pd.DataFrame, rating_system: AdvancedRatingSystem) -> pd.DataFrame:
        """Crée des features dérivées des statistiques"""
        
        # Features de ratio
        df['goal_ratio_home'] = df['home_goals_for'] / (df['home_goals_against'] + 1)
        df['goal_ratio_away'] = df['away_goals_for'] / (df['away_goals_against'] + 1)
        df['possession_diff'] = df['home_possession'] - df['away_possession']
        
        # Features d'efficacité
        df['shooting_efficiency_home'] = df['home_shots_on_goal'] / (df['home_total_shots'] + 1)
        df['shooting_efficiency_away'] = df['away_shots_on_goal'] / (df['away_total_shots'] + 1)
        
        # Features de pression
        df['pressure_ratio'] = (df['home_corner_kicks'] + df['home_free_kicks']) / \
                              (df['away_corner_kicks'] + df['away_free_kicks'] + 1)
        
        # Features de rating Elo/Glicko
        for idx, row in df.iterrows():
            if pd.notnull(row['home_id']) and pd.notnull(row['away_id']):
                rating_features = rating_system.get_match_features(
                    int(row['home_id']), 
                    int(row['away_id'])
                )
                for key, value in rating_features.items():
                    df.loc[idx, f'rating_{key}'] = value
        
        return df

# =============================================================================
# MODÈLES DE MACHINE LEARNING AVANCÉS
# =============================================================================

class AdvancedBettingModel:
    """Modèle de prédiction avancé avec calibration et ensembling"""
    
    def __init__(self, sport: str):
        self.sport = sport
        self.models = {}
        self.scalers = {}
        self.calibration_model = None
        self.feature_importance = {}
        
        # Configuration des modèles
        self.model_configs = {
            'lgbm': {
                'classifier': LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.01,
                    max_depth=7,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'regressor': LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.01,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            },
            'xgb': {
                'classifier': xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.01,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            }
        }
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'home_win') -> Tuple:
        """Prépare les features et la target"""
        
        # Features de base
        base_features = [
            'elo_diff', 'form_diff', 'home_advantage', 
            'off_rating_diff', 'def_rating_diff', 'net_rating_diff'
        ]
        
        # Features avancées
        advanced_features = [
            'goal_ratio_home', 'goal_ratio_away', 'possession_diff',
            'shooting_efficiency_home', 'shooting_efficiency_away',
            'pressure_ratio', 'momentum', 'days_since_last_game_home',
            'days_since_last_game_away', 'is_weekend'
        ]
        
        all_features = base_features + advanced_features
        available_features = [f for f in all_features if f in df.columns]
        
        # Séparer features et target
        X = df[available_features].copy()
        y = df[target_col] if target_col in df.columns else None
        
        # Gestion des valeurs manquantes
        X = X.fillna(X.median())
        
        # Normalisation
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X_scaled = self.scalers['scaler'].fit_transform(X)
        else:
            X_scaled = self.scalers['scaler'].transform(X)
        
        return X_scaled, y, available_features
    
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Entraîne un ensemble de modèles"""
        
        models_trained = {}
        
        for model_name, config in self.model_configs.items():
            st.info(f"Entraînement du modèle {model_name}...")
            
            # Modèle de classification
            model = config['classifier']
            
            # Validation croisée temporelle
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=tscv, scoring='roc_auc', n_jobs=-1
            )
            
            # Entraînement final
            model.fit(X_train, y_train)
            models_trained[model_name] = model
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            
            st.success(f"{model_name}: AUC CV = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        self.models = models_trained
        
        # Calibration isotonique pour les probabilités
        self._calibrate_probabilities(X_train, y_train)
    
    def _calibrate_probabilities(self, X, y):
        """Calibre les probabilités avec Platt Scaling"""
        from sklearn.calibration import CalibratedClassifierCV
        
        if self.models:
            base_model = list(self.models.values())[0]
            self.calibration_model = CalibratedClassifierCV(
                base_model, method='isotonic', cv=3
            )
            self.calibration_model.fit(X, y)
    
    def predict_proba_ensemble(self, X):
        """Prédit avec l'ensemble des modèles"""
        
        if not self.models:
            raise ValueError("Modèles non entraînés")
        
        predictions = []
        weights = {'lgbm': 0.6, 'xgb': 0.4}  # Poids pour l'ensembling
        
        for model_name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred * weights.get(model_name, 0.5))
        
        # Moyenne pondérée
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Calibration si disponible
        if self.calibration_model is not None:
            ensemble_pred = self.calibration_model.predict_proba(X)[:, 1]
        
        return ensemble_pred
    
    def calculate_confidence_intervals(self, X, n_bootstrap=100):
        """Calcule les intervalles de confiance par bootstrap"""
        
        if not self.models:
            return np.zeros(len(X)), np.zeros(len(X))
        
        predictions_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Rééchantillonnage bootstrap
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            
            # Prédiction avec l'ensemble
            pred = self.predict_proba_ensemble(X_boot)
            predictions_bootstrap.append(pred)
        
        predictions_bootstrap = np.array(predictions_bootstrap)
        
        # Calcul des intervalles
        mean_pred = np.mean(predictions_bootstrap, axis=0)
        std_pred = np.std(predictions_bootstrap, axis=0)
        
        # Intervalle de confiance 95%
        ci_lower = mean_pred - 1.96 * std_pred
        ci_upper = mean_pred + 1.96 * std_pred
        
        return ci_lower, ci_upper

# =============================================================================
# VALUE BET DETECTION & BANKROLL MANAGEMENT
# =============================================================================

class AdvancedValueBetDetector:
    """Détection avancée de value bets avec gestion de bankroll"""
    
    def __init__(self, bankroll: float = 10000.0):
        self.bankroll = bankroll
        self.bet_history = []
        self.current_stake = 0.0
        
    @staticmethod
    def calculate_implied_probability(odds: float) -> float:
        """Calcule la probabilité implicite d'une cote"""
        return 1 / odds if odds > 1 else 0.0
    
    @staticmethod
    def calculate_edge(model_prob: float, odds: float) -> float:
        """Calcule l'edge (valeur attendue)"""
        implied_prob = AdvancedValueBetDetector.calculate_implied_probability(odds)
        return model_prob * odds - 1 if model_prob > 0 else -1
    
    @staticmethod
    def calculate_kelly_stake(edge: float, odds: float, fraction: float = Config.KELLY_FRACTION) -> float:
        """Calcule la mise selon le critère de Kelly"""
        if edge <= 0 or odds <= 1:
            return 0.0
        
        # Formule de Kelly: f* = (bp - q) / b
        b = odds - 1
        p = edge / b + (1 / odds)  # Probabilité réelle estimée
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Limiter à 25%
        
        return kelly_fraction * fraction
    
    def evaluate_bet(self, match_info: Dict, model_prob: float, 
                    home_odds: float, away_odds: float) -> Optional[Dict]:
        """Évalue un pari potentiel"""
        
        # Déterminer le côté avec le meilleur edge
        home_edge = self.calculate_edge(model_prob, home_odds)
        away_edge = self.calculate_edge(1 - model_prob, away_odds)
        
        if home_edge > away_edge and home_edge > Config.MIN_EDGE_THRESHOLD:
            best_side = 'home'
            best_edge = home_edge
            best_odds = home_odds
            best_prob = model_prob
        elif away_edge > home_edge and away_edge > Config.MIN_EDGE_THRESHOLD:
            best_side = 'away'
            best_edge = away_edge
            best_odds = away_odds
            best_prob = 1 - model_prob
        else:
            return None  # Pas de value bet
        
        # Calculer la mise Kelly
        kelly_fraction = self.calculate_kelly_stake(best_edge, best_odds)
        stake_amount = kelly_fraction * self.bankroll
        
        # Limiter la mise
        max_stake = Config.MAX_STAKE_PERCENT * self.bankroll
        stake_amount = min(stake_amount, max_stake)
        
        # Calculer l'EV (Expected Value)
        ev = stake_amount * best_edge
        
        # Calculer le ROI attendu
        expected_roi = best_edge * 100
        
        return {
            'match': f"{match_info.get('home_team')} vs {match_info.get('away_team')}",
            'league': match_info.get('league', 'Unknown'),
            'date': match_info.get('date'),
            'side': best_side,
            'model_probability': best_prob,
            'bookmaker_odds': best_odds,
            'implied_probability': self.calculate_implied_probability(best_odds),
            'edge': best_edge,
            'edge_percentage': best_edge * 100,
            'kelly_fraction': kelly_fraction,
            'recommended_stake': stake_amount,
            'expected_value': ev,
            'expected_roi': expected_roi,
            'confidence': min(best_prob * 1.2, 0.95),  # Score de confiance
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# INTERFACE STREAMLIT AVANCÉE
# =============================================================================

def main():
    st.set_page_config(
        page_title="Système de Paris Sportifs IA Avancé",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .stAlert { padding: 20px; border-radius: 10px; }
    .stButton > button { width: 100%; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎯 Système de Paris Sportifs IA - Version Professionnelle")
    st.markdown("**IA Avancée • Elo/Glicko • Machine Learning • Gestion de Bankroll**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sélection du sport
        sport = st.selectbox(
            "Sport",
            ["Football", "Basketball (NBA)", "Tennis", "Baseball", "Rugby"],
            index=0
        )
        
        # Bankroll initial
        bankroll = st.number_input(
            "Bankroll Initial (€)",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=500.0
        )
        
        # Paramètres de risque
        st.subheader("📊 Paramètres de Risque")
        kelly_fraction = st.slider(
            "Fraction de Kelly", 
            min_value=0.1, 
            max_value=1.0, 
            value=Config.KELLY_FRACTION,
            help="Pourcentage du Kelly full à utiliser"
        )
        
        min_edge = st.slider(
            "Edge Minimum (%)",
            min_value=1.0,
            max_value=20.0,
            value=Config.MIN_EDGE_THRESHOLD * 100,
            step=0.5
        ) / 100
        
        # Options avancées
        with st.expander("Options Avancées"):
            use_live_odds = st.checkbox("Utiliser cotes en direct", value=True)
            include_advanced_stats = st.checkbox("Stats avancées", value=True)
            enable_backtesting = st.checkbox("Backtesting", value=False)
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rafraîchir les Données", type="primary"):
                st.rerun()
        with col2:
            if st.button("📊 Analyser", type="secondary"):
                st.session_state.analyze = True
    
    # Initialisation des systèmes
    if 'rating_system' not in st.session_state:
        st.session_state.rating_system = AdvancedRatingSystem(sport.lower())
    
    if 'data_collector' not in st.session_state:
        st.session_state.data_collector = AdvancedDataCollector()
    
    if 'betting_model' not in st.session_state:
        st.session_state.betting_model = AdvancedBettingModel(sport)
    
    if 'value_detector' not in st.session_state:
        st.session_state.value_detector = AdvancedValueBetDetector(bankroll)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Dashboard", 
        "🎯 Value Bets", 
        "🤖 Modèle IA", 
        "💰 Bankroll", 
        "📊 Backtest"
    ])
    
    with tab1:
        # Dashboard principal
        st.header("Dashboard de Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("
