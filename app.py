# =============================================================================
# PRONOSTIQUEUR MULTI-SPORTS PROFESSIONNEL - Value Bets Advanced
# Version: 3.1 Ultra-Précise - Optimisée pour les paris value
# Compatible Streamlit Cloud (pas de xgboost requis)
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

# Librairies ML disponibles sur Streamlit Cloud
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

# =============================================================================
# CONFIGURATION AVANCÉE
# =============================================================================

class Config:
    """Configuration globale ultra-précise"""
    
    ELO_K_FACTOR = 32
    ELO_HOME_ADVANTAGE = 70
    ELO_BASE_RATING = 1500
    FEATURE_LAG_DAYS = 30
    MIN_TRAINING_SAMPLES = 100  # Réduit pour éviter erreurs
    KELLY_FRACTION = 0.25
    MAX_STAKE_PERCENT = 0.02
    MIN_EDGE_THRESHOLD = 0.02
    CONFIDENCE_THRESHOLD = 0.65
    
    CACHE_TTL_BASIC = 1800
    CACHE_TTL_ADVANCED = 3600
    CACHE_TTL_LIVE = 300
    
    # Clés API - à configurer dans Streamlit Secrets
    API_SPORTS_KEY = st.secrets.get("API_SPORTS_KEY", "demo_key")
    ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "demo_key")
    
    NBA_SEASON = "2024-25"
    FOOTBALL_SEASON = 2024

# =============================================================================
# SYSTÈMES DE RATING ELO/GLICKO
# =============================================================================

class AdvancedRatingSystem:
    """Système de rating hybride Elo/Glicko"""
    
    def __init__(self, sport: str, decay_factor: float = 0.95):
        self.sport = sport.lower()
        self.decay_factor = decay_factor
        self.ratings = {}
        self.history = {}
        
        self.sport_factors = {
            'basketball': {'k_factor': 24, 'home_adv': 100, 'margin_factor': 2.2},
            'football': {'k_factor': 32, 'home_adv': 70, 'margin_factor': 1.8},
            'tennis': {'k_factor': 40, 'home_adv': 0, 'margin_factor': 1.5},
            'baseball': {'k_factor': 28, 'home_adv': 50, 'margin_factor': 1.6},
            'rugby': {'k_factor': 30, 'home_adv': 60, 'margin_factor': 1.7},
        }
        
        self.factors = self.sport_factors.get(self.sport, self.sport_factors['football'])
    
    def get_initial_rating(self, team_id: int) -> Dict:
        return {
            'elo': Config.ELO_BASE_RATING,
            'glicko_rating': Config.ELO_BASE_RATING,
            'glicko_rd': 350,
            'volatility': 0.06,
            'games_played': 0,
            'last_game_date': None,
            'streak': 0,
            'form': 0.5,
            'offensive_rating': 100,
            'defensive_rating': 100
        }
    
    def margin_of_victory_multiplier(self, point_diff: float, rating_diff: float) -> float:
        return np.log(abs(point_diff) + 1) * (2.2 / (rating_diff * 0.001 + 2.2))
    
    def update_ratings(self, home_id: int, away_id: int, 
                      home_score: int, away_score: int, 
                      game_date: date, venue: str = 'home'):
        
        if home_id not in self.ratings:
            self.ratings[home_id] = self.get_initial_rating(home_id)
        if away_id not in self.ratings:
            self.ratings[away_id] = self.get_initial_rating(away_id)
        
        home_rating = self.ratings[home_id]
        away_rating = self.ratings[away_id]
        
        self.apply_time_decay(home_rating, game_date)
        self.apply_time_decay(away_rating, game_date)
        
        if home_score > away_score:
            actual_score = 1.0
        elif home_score < away_score:
            actual_score = 0.0
        else:
            actual_score = 0.5
        
        home_advantage = self.factors['home_adv'] if venue == 'home' else 0
        
        expected_home = 1 / (1 + 10 ** ((away_rating['elo'] - home_rating['elo'] - home_advantage) / 400))
        
        point_diff = home_score - away_score
        mov_multiplier = self.margin_of_victory_multiplier(
            point_diff, 
            home_rating['elo'] - away_rating['elo']
        )
        
        elo_change = self.factors['k_factor'] * mov_multiplier * (actual_score - expected_home)
        home_rating['elo'] += elo_change
        away_rating['elo'] -= elo_change
        
        self.update_form_and_streak(home_rating, away_rating, actual_score)
        self.update_offensive_defensive_ratings(
            home_rating, away_rating, home_score, away_score
        )
        
        match_key = f"{game_date}_{home_id}_{away_id}"
        self.history[match_key] = {
            'home_elo': home_rating['elo'],
            'away_elo': away_rating['elo'],
            'home_score': home_score,
            'away_score': away_score,
            'result': actual_score
        }
        
        home_rating['last_game_date'] = game_date
        away_rating['last_game_date'] = game_date
        home_rating['games_played'] += 1
        away_rating['games_played'] += 1
    
    def apply_time_decay(self, rating: Dict, current_date: date):
        if rating['last_game_date']:
            days_since = (current_date - rating['last_game_date']).days
            decay = self.decay_factor ** (days_since / 30)
            rating['elo'] = rating['elo'] * decay + Config.ELO_BASE_RATING * (1 - decay)
            rating['form'] *= decay
    
    def update_form_and_streak(self, home_rating: Dict, away_rating: Dict, result: float):
        home_rating['form'] = home_rating['form'] * 0.9 + result * 0.1
        away_rating['form'] = away_rating['form'] * 0.9 + (1 - result) * 0.1
        
        if result == 1:
            home_rating['streak'] = max(1, home_rating['streak'] + 1)
            away_rating['streak'] = min(-1, away_rating['streak'] - 1)
        elif result == 0:
            home_rating['streak'] = min(-1, home_rating['streak'] - 1)
            away_rating['streak'] = max(1, away_rating['streak'] + 1)
    
    def update_offensive_defensive_ratings(self, home_rating: Dict, away_rating: Dict,
                                          home_score: int, away_score: int):
        alpha = 0.15
        
        home_rating['offensive_rating'] = (home_rating['offensive_rating'] * (1 - alpha) + 
                                          home_score * alpha)
        home_rating['defensive_rating'] = (home_rating['defensive_rating'] * (1 - alpha) + 
                                          away_score * alpha)
        
        away_rating['offensive_rating'] = (away_rating['offensive_rating'] * (1 - alpha) + 
                                          away_score * alpha)
        away_rating['defensive_rating'] = (away_rating['defensive_rating'] * (1 - alpha) + 
                                          home_score * alpha)
    
    def get_match_features(self, home_id: int, away_id: int, venue: str = 'home') -> Dict:
        if home_id not in self.ratings or away_id not in self.ratings:
            return {}
        
        home = self.ratings[home_id]
        away = self.ratings[away_id]
        
        return {
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

# =============================================================================
# COLLECTE DE DONNÉES AVANCÉE (avec fallback)
# =============================================================================

class AdvancedDataCollector:
    """Collecteur de données multi-sources avec fallback"""
    
    def __init__(self):
        self.sources = {
            'api-football': 'https://v3.football.api-sports.io',
            'odds-api': 'https://api.the-odds-api.com/v4',
        }
        self.demo_mode = Config.API_SPORTS_KEY == "demo_key"
    
    @st.cache_data(ttl=Config.CACHE_TTL_BASIC)
    def get_football_fixtures(self, league_id: int, season: int, 
                             date_from: str, date_to: str) -> pd.DataFrame:
        
        if self.demo_mode:
            # Mode démo avec données simulées
            return self.get_demo_football_data(date_from, date_to)
        
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
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            fixtures = []
            for fixture in data.get('response', []):
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
                    'home_score': fixture['goals']['home'] if fixture['goals']['home'] is not None else 0,
                    'away_score': fixture['goals']['away'] if fixture['goals']['away'] is not None else 0,
                })
            
            return pd.DataFrame(fixtures)
            
        except Exception as e:
            st.warning(f"API Football non disponible: {e}. Utilisation mode démo.")
            return self.get_demo_football_data(date_from, date_to)
    
    def get_demo_football_data(self, date_from: str, date_to: str) -> pd.DataFrame:
        """Génère des données de démo réalistes"""
        teams = [
            ("Manchester City", 50, 85),
            ("Liverpool", 51, 84),
            ("Arsenal", 52, 83),
            ("Chelsea", 53, 78),
            ("Tottenham", 54, 77),
            ("Manchester United", 55, 76),
            ("Real Madrid", 56, 88),
            ("Barcelona", 57, 86),
            ("Bayern Munich", 58, 87),
            ("PSG", 59, 85),
        ]
        
        fixtures = []
        start_date = pd.to_datetime(date_from)
        end_date = pd.to_datetime(date_to)
        
        current_date = start_date
        while current_date <= end_date:
            # Générer 2-3 matchs par jour
            for _ in range(np.random.randint(2, 4)):
                home_idx, away_idx = np.random.choice(len(teams), 2, replace=False)
                home_team, home_id, home_elo = teams[home_idx]
                away_team, away_id, away_elo = teams[away_idx]
                
                # Calculer la probabilité de victoire à domicile basée sur Elo
                home_adv = 70
                expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - home_adv) / 400))
                
                # Simuler le résultat
                home_wins = np.random.binomial(1, expected_home)
                home_score = np.random.poisson(1.8) if home_wins else np.random.poisson(1.2)
                away_score = np.random.poisson(1.2) if home_wins else np.random.poisson(1.8)
                
                fixtures.append({
                    'fixture_id': np.random.randint(10000, 99999),
                    'date': current_date.strftime('%Y-%m-%d'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_id': home_id,
                    'away_id': away_id,
                    'league_id': 39,  # Premier League
                    'season': Config.FOOTBALL_SEASON,
                    'status': 'FT',
                    'home_score': home_score,
                    'away_score': away_score,
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(fixtures)
    
    def get_odds_data(self, sport: str = 'soccer_epl'):
        """Récupère ou simule des cotes"""
        if Config.ODDS_API_KEY == "demo_key" or self.demo_mode:
            return self.get_demo_odds_data()
        
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
            params = {
                'apiKey': Config.ODDS_API_KEY,
                'regions': 'eu',
                'markets': 'h2h',
                'oddsFormat': 'decimal'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            st.warning(f"API Odds non disponible: {e}")
            return self.get_demo_odds_data()
    
    def get_demo_odds_data(self):
        """Génère des cotes de démo réalistes"""
        matches = [
            {"home_team": "Manchester City", "away_team": "Liverpool", "home_odds": 1.85, "away_odds": 4.00, "draw_odds": 3.60},
            {"home_team": "Arsenal", "away_team": "Chelsea", "home_odds": 2.10, "away_odds": 3.50, "draw_odds": 3.30},
            {"home_team": "Real Madrid", "away_team": "Barcelona", "home_odds": 2.30, "away_odds": 3.00, "draw_odds": 3.40},
            {"home_team": "Bayern Munich", "away_team": "Borussia Dortmund", "home_odds": 1.65, "away_odds": 5.00, "draw_odds": 4.00},
            {"home_team": "PSG", "away_team": "Marseille", "home_odds": 1.50, "away_odds": 6.00, "draw_odds": 4.20},
        ]
        
        return matches

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class AdvancedFeatureEngineer:
    """Génération de features avancées"""
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['hour'] = df['date'].dt.hour
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    @staticmethod
    def calculate_form(df: pd.DataFrame, team_col: str, window: int = 5) -> pd.DataFrame:
        """Calcule la forme sur les N derniers matchs"""
        df = df.sort_values('date')
        
        for team_id in df[team_col].unique():
            team_matches = df[df[team_col] == team_id].copy()
            
            # Calculer les points (3 pour victoire, 1 pour nul, 0 pour défaite)
            team_matches['points'] = team_matches.apply(
                lambda row: 3 if row['home_score'] > row['away_score'] else 
                           1 if row['home_score'] == row['away_score'] else 0,
                axis=1
            )
            
            # Forme sur fenêtre glissante
            team_matches[f'{team_col}_form_{window}'] = team_matches['points'].rolling(window, min_periods=1).mean() / 3
            
            # Mettre à jour le DataFrame original
            df.loc[team_matches.index, f'{team_col}_form_{window}'] = team_matches[f'{team_col}_form_{window}']
        
        return df

# =============================================================================
# MODÈLES DE MACHINE LEARNING (sans xgboost)
# =============================================================================

class AdvancedBettingModel:
    """Modèle de prédiction avancé avec ensembling"""
    
    def __init__(self, sport: str):
        self.sport = sport
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Configuration des modèles disponibles
        self.model_configs = {
            'lgbm': LGBMClassifier(
                n_estimators=300,
                learning_rate=0.01,
                max_depth=6,
                num_leaves=31,
                min_child_samples=15,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                force_row_wise=True  # Évite les warnings
            ),
            'gbc': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'logreg': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
        }
    
    def prepare_features(self, df: pd.DataFrame, rating_system: AdvancedRatingSystem) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les features pour l'entraînement"""
        
        features_list = []
        
        for idx, row in df.iterrows():
            features = {}
            
            # Features de rating Elo
            rating_features = rating_system.get_match_features(
                row['home_id'], 
                row['away_id']
            )
            features.update(rating_features)
            
            # Features temporelles
            features['day_of_week'] = row.get('day_of_week', 0)
            features['is_weekend'] = row.get('is_weekend', 0)
            
            # Features de forme
            features['home_form_5'] = row.get('home_id_form_5', 0.5)
            features['away_form_5'] = row.get('away_id_form_5', 0.5)
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()
        
        # Gérer les valeurs manquantes
        features_df = features_df.fillna(0)
        
        # Normalisation
        if len(features_df) > 0:
            features_scaled = self.scaler.fit_transform(features_df)
        else:
            features_scaled = np.array([])
        
        # Target (victoire à domicile)
        if 'home_score' in df.columns and 'away_score' in df.columns:
            target = (df['home_score'] > df['away_score']).astype(int).values
        else:
            target = np.array([])
        
        return features_scaled, target
    
    def train(self, X_train, y_train):
        """Entraîne l'ensemble de modèles"""
        
        if len(X_train) < Config.MIN_TRAINING_SAMPLES:
            st.warning(f"Données d'entraînement insuffisantes ({len(X_train)} échantillons). Utilisation de modèles par défaut.")
            return self._create_default_models()
        
        st.info(f"Entraînement sur {len(X_train)} échantillons...")
        
        trained_models = {}
        
        for model_name, model in self.model_configs.items():
            try:
                st.write(f"  • Entraînement {model_name}...")
                
                # Validation croisée
                tscv = TimeSeriesSplit(n_splits=min(5, len(X_train)//10))
                
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=tscv, scoring='roc_auc',
                    n_jobs=1  # Évite les problèmes de mémoire
                )
                
                # Entraînement final
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                
                st.success(f"    {model_name}: AUC = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
                
            except Exception as e:
                st.warning(f"    Erreur avec {model_name}: {str(e)[:100]}")
        
        if not trained_models:
            st.error("Aucun modèle n'a pu être entraîné. Utilisation de modèles par défaut.")
            trained_models = self._create_default_models()
        
        self.models = trained_models
        
        # Calibration isotonique sur le meilleur modèle
        if self.models:
            best_model_name = list(self.models.keys())[0]
            best_model = self.models[best_model_name]
            
            try:
                self.calibration_model = CalibratedClassifierCV(
                    best_model, method='isotonic', cv=3
                )
                self.calibration_model.fit(X_train, y_train)
                st.success("✓ Calibration des probabilités terminée")
            except:
                self.calibration_model = None
                st.warning("Calibration non disponible")
    
    def _create_default_models(self):
        """Crée des modèles par défaut simples"""
        return {
            'logreg': LogisticRegression(random_state=42),
            'rf': RandomForestClassifier(n_estimators=50, random_state=42)
        }
    
    def predict_proba(self, X):
        """Prédit avec l'ensemble des modèles"""
        
        if not self.models:
            raise ValueError("Modèles non entraînés")
        
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)
            except:
                # Si un modèle échoue, ignorer
                continue
        
        if not predictions:
            # Fallback: probabilité de 0.5
            return np.ones(len(X)) * 0.5
        
        # Moyenne des prédictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Calibration si disponible
        if hasattr(self, 'calibration_model') and self.calibration_model is not None:
            try:
                ensemble_pred = self.calibration_model.predict_proba(X)[:, 1]
            except:
                pass
        
        return ensemble_pred
    
    def get_feature_importance(self):
        """Retourne l'importance des features"""
        if 'lgbm' in self.models:
            model = self.models['lgbm']
            if hasattr(model, 'feature_importances_'):
                return dict(zip(self.feature_names, model.feature_importances_))
        
        return {name: 1.0/len(self.feature_names) for name in self.feature_names}

# =============================================================================
# VALUE BET DETECTION
# =============================================================================

class ValueBetAnalyzer:
    """Analyseur de value bets"""
    
    @staticmethod
    def calculate_implied_probability(odds: float) -> float:
        """Calcule la probabilité implicite"""
        if odds <= 1:
            return 0.0
        return 1.0 / odds
    
    @staticmethod
    def calculate_edge(model_prob: float, odds: float) -> float:
        """Calcule l'edge (Expected Value)"""
        implied_prob = ValueBetAnalyzer.calculate_implied_probability(odds)
        if model_prob <= 0:
            return -1.0
        return (model_prob * odds) - 1.0
    
    @staticmethod
    def calculate_kelly_stake(edge: float, odds: float, bankroll: float, 
                             fraction: float = Config.KELLY_FRACTION) -> float:
        """Calcule la mise selon Kelly"""
        if edge <= 0 or odds <= 1:
            return 0.0
        
        b = odds - 1
        p = (edge / b) + (1 / odds)
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Limiter
        
        stake = kelly_fraction * fraction * bankroll
        
        # Limiter à 2% maximum
        max_stake = Config.MAX_STAKE_PERCENT * bankroll
        return min(stake, max_stake)
    
    @staticmethod
    def analyze_match(match_data: Dict, model_prob: float, 
                     home_odds: float, away_odds: float, 
                     draw_odds: Optional[float] = None,
                     bankroll: float = 10000.0) -> Optional[Dict]:
        """Analyse un match pour trouver des value bets"""
        
        # Analyse victoire domicile
        home_edge = ValueBetAnalyzer.calculate_edge(model_prob, home_odds)
        home_implied = ValueBetAnalyzer.calculate_implied_probability(home_odds)
        
        # Analyse victoire extérieur
        away_prob = 1 - model_prob
        away_edge = ValueBetAnalyzer.calculate_edge(away_prob, away_odds)
        away_implied = ValueBetAnalyzer.calculate_implied_probability(away_odds)
        
        # Trouver le meilleur edge
        best_edge = max(home_edge, away_edge)
        
        if best_edge < Config.MIN_EDGE_THRESHOLD:
            return None
        
        # Déterminer le meilleur pari
        if home_edge >= away_edge:
            best_side = 'Home'
            best_odds = home_odds
            best_prob = model_prob
            best_edge_value = home_edge
            best_implied = home_implied
        else:
            best_side = 'Away'
            best_odds = away_odds
            best_prob = away_prob
            best_edge_value = away_edge
            best_implied = away_implied
        
        # Calculer la mise Kelly
        kelly_stake = ValueBetAnalyzer.calculate_kelly_stake(
            best_edge_value, best_odds, bankroll
        )
        
        # Expected Value
        expected_value = kelly_stake * best_edge_value
        
        # Confiance
        confidence = min(best_prob * 1.5, 0.95)
        
        # ROI attendu
        expected_roi = best_edge_value * 100
        
        return {
            'match': f"{match_data.get('home_team', 'Home')} vs {match_data.get('away_team', 'Away')}",
            'date': match_data.get('date', 'N/A'),
            'league': match_data.get('league', 'Unknown'),
            'best_bet': best_side,
            'model_probability': best_prob,
            'bookmaker_odds': best_odds,
            'implied_probability': best_implied,
            'edge': best_edge_value,
            'edge_percentage': best_edge_value * 100,
            'confidence': confidence,
            'kelly_stake': kelly_stake,
            'expected_value': expected_value,
            'expected_roi': expected_roi,
            'value_score': best_edge_value * confidence * 100,  # Score composite
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def main():
    st.set_page_config(
        page_title="Système de Paris Sportifs IA",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">⚽ Système de Paris Sportifs IA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">IA Avancée • Elo Rating • Value Bets • Gestion Bankroll</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sélection du sport
        sport_options = {
            "Football": "football",
            "Basketball (NBA)": "basketball", 
            "Tennis": "tennis",
            "Baseball": "baseball",
            "Rugby": "rugby"
        }
        
        selected_sport = st.selectbox(
            "Choisir un sport",
            list(sport_options.keys()),
            index=0
        )
        
        sport_key = sport_options[selected_sport]
        
        # Bankroll
        bankroll = st.number_input(
            "💰 Bankroll (€)",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=500.0
        )
        
        # Paramètres de risque
        st.subheader("📊 Paramètres Risque")
        
        kelly_fraction = st.slider(
            "Fraction de Kelly", 
            min_value=0.1,
            max_value=1.0,
            value=Config.KELLY_FRACTION,
            step=0.05,
            help="Pourcentage du Kelly full à utiliser (plus conservateur)"
        )
        
        min_edge = st.slider(
            "Edge minimum (%)",
            min_value=1.0,
            max_value=15.0,
            value=Config.MIN_EDGE_THRESHOLD * 100,
            step=0.5
        ) / 100.0
        
        # Options
        with st.expander("🔧 Options avancées"):
            demo_mode = st.checkbox("Mode démo (sans API)", value=True)
            auto_refresh = st.checkbox("Rafraîchissement auto", value=False)
            
            if not demo_mode:
                st.info("Configurer les clés API dans Streamlit Secrets")
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Analyser", type="primary"):
                st.session_state.analyze = True
        with col2:
            if st.button("🧹 Réinitialiser"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Info
        st.divider()
        st.info("**ℹ️ Mode démo activé**\nLes données sont simulées pour la démonstration.")
    
    # Initialisation
    if 'rating_system' not in st.session_state:
        st.session_state.rating_system = AdvancedRatingSystem(sport_key)
    
    if 'data_collector' not in st.session_state:
        st.session_state.data_collector = AdvancedDataCollector()
    
    if 'betting_model' not in st.session_state:
        st.session_state.betting_model = AdvancedBettingModel(sport_key)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🎯 Value Bets", 
        "🤖 Modèles IA", 
        "📈 Performance"
    ])
    
    with tab1:
        st.header("📊 Dashboard Principal")
        
        # Métriques principales
        col1, col2, col3, col
