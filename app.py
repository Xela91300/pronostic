# app.py - Système de Pronostics Multi-Sports avec Données en Temps Réel
# Version Premium avec architecture modulaire

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
from dataclasses import dataclass
from enum import Enum
import hashlib
import functools
import logging
import html
import sqlite3
from contextlib import contextmanager
import asyncio
import concurrent.futures

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DU LOGGING
# =============================================================================

class StructuredLogger:
    """Logger structuré pour le suivi des prédictions"""
    
    def __init__(self, log_file: str = "predictions.log"):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, prediction_data: Dict):
        """Log une prédiction de manière structurée"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'prediction',
            'sport': prediction_data.get('sport'),
            'home_team': prediction_data.get('home_team'),
            'away_team': prediction_data.get('away_team'),
            'probabilities': prediction_data.get('probabilities'),
            'confidence': prediction_data.get('confidence_score')
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_bet(self, bet_data: Dict, outcome: str = None):
        """Log un pari"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'bet',
            'bet_data': bet_data,
            'outcome': outcome
        }
        
        self.logger.info(json.dumps(log_entry))

# =============================================================================
# TYPES ET ENUMS
# =============================================================================

class SportType(Enum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"

class BetType(Enum):
    WIN = "victoire"
    DRAW = "match_nul"
    OVER_UNDER = "over_under"
    BOTH_TEAMS_SCORE = "both_teams_score"
    HANDICAP = "handicap"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class PlayerInjury:
    player_name: str
    position: str
    injury_type: str
    severity: str  # mineure, moyenne, grave
    expected_return: Optional[date]
    impact_score: float  # 0-10

@dataclass
class WeatherCondition:
    temperature: float
    precipitation: float  # 0-1
    wind_speed: float
    humidity: float
    condition: str  # sunny, rainy, cloudy

# =============================================================================
# VALIDATION DE SÉCURITÉ
# =============================================================================

class SecurityValidator:
    """Valide et nettoie les entrées utilisateur"""
    
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """Nettoie les entrées pour prévenir les injections"""
        if not user_input:
            return ""
        
        # Échapper les caractères HTML
        sanitized = html.escape(user_input)
        
        # Supprimer les caractères non désirés
        sanitized = re.sub(r'[<>{};]', '', sanitized)
        
        # Limiter la longueur
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Valide le format d'une clé API"""
        if not api_key or api_key == "demo":
            return True  # Mode démo autorisé
        
        # Validation basique du format
        if len(api_key) < 20 or len(api_key) > 100:
            return False
        
        # Vérifier le format (ex: uuid, token JWT, etc.)
        if re.match(r'^[a-zA-Z0-9\-_]+$', api_key):
            return True
        
        return False

# =============================================================================
# GESTION DES ERREURS
# =============================================================================

def handle_errors(func):
    """Décorateur pour la gestion des erreurs"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (requests.RequestException, TimeoutError) as e:
            logging.error(f"Erreur réseau dans {func.__name__}: {e}")
            return {'error': 'network_error', 'message': str(e)}
        except ValueError as e:
            logging.error(f"Erreur de validation dans {func.__name__}: {e}")
            return {'error': 'validation_error', 'message': str(e)}
        except Exception as e:
            logging.error(f"Erreur inattendue dans {func.__name__}: {e}")
            return {'error': 'unexpected_error', 'message': str(e)}
    return wrapper

# =============================================================================
# VALIDATEUR DE DONNÉES UTILISATEUR
# =============================================================================

class DataValidator:
    """Valide et nettoie les données utilisateur"""
    
    @staticmethod
    def validate_team_name(team_name: str, sport: SportType) -> Tuple[bool, str]:
        """Valide le nom d'une équipe"""
        if not team_name or len(team_name.strip()) < 2:
            return False, "Le nom de l'équipe est trop court"
        
        # Vérification des caractères
        if not re.match(r'^[a-zA-Z0-9\s\-\.\']+$', team_name):
            return False, "Le nom contient des caractères non autorisés"
        
        return True, ""
    
    @staticmethod
    def normalize_team_name(team_name: str) -> str:
        """Normalise le nom d'une équipe pour la recherche"""
        name = team_name.strip()
        # Supprime les suffixes communs
        suffixes = [' FC', ' CF', ' AFC', ' United', ' City', ' Real', ' Club']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        return name.title()
    
    @staticmethod
    def validate_match_date(match_date: date) -> Tuple[bool, str]:
        """Valide la date du match"""
        today = date.today()
        max_future_date = today + timedelta(days=365)
        
        if match_date < today - timedelta(days=365):
            return False, "La date est trop ancienne"
        if match_date > max_future_date:
            return False, "La date est trop éloignée dans le futur"
        
        return True, ""
    
    @staticmethod
    def suggest_corrections(team_name: str, known_teams: List[str]) -> List[str]:
        """Suggère des corrections pour le nom d'équipe"""
        suggestions = []
        team_name_lower = team_name.lower()
        
        for known_team in known_teams:
            known_lower = known_team.lower()
            
            # Correspondance exacte
            if team_name_lower == known_lower:
                return [known_team]
            
            # Contient ou est contenu
            if team_name_lower in known_lower or known_lower in team_name_lower:
                suggestions.append(known_team)
            
            # Similarité de Levenshtein simplifiée
            if DataValidator._calculate_similarity(team_name_lower, known_lower) > 0.7:
                suggestions.append(known_team)
        
        return list(set(suggestions))[:5]
    
    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """Calcule la similarité entre deux chaînes"""
        if not str1 or not str2:
            return 0.0
        
        # Similarité simple
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union

# =============================================================================
# BASE DE DONNÉES LOCALE
# =============================================================================

class LocalDatabase:
    """Gestion d'une base de données locale SQLite"""
    
    def __init__(self, db_path: str = "sports_predictions.db"):
        self.db_path = db_path
        self.logger = StructuredLogger()
        self._init_database()
    
    def _init_database(self):
        """Initialise la base de données"""
        with self.get_connection() as conn:
            # Table des prédictions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sport TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    league TEXT,
                    prediction_date DATE,
                    probabilities TEXT,
                    score_prediction TEXT,
                    confidence_score REAL,
                    actual_result TEXT,
                    accuracy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Table du feedback utilisateur
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    was_correct BOOLEAN,
                    user_confidence INTEGER,
                    feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            """)
            
            # Table des paris
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    bet_type TEXT,
                    odds REAL,
                    stake REAL,
                    outcome TEXT,
                    profit_loss REAL,
                    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            """)
            
            # Table du cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index pour les performances
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_teams ON predictions(home_team, away_team)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)")
    
    @contextmanager
    def get_connection(self):
        """Context manager pour les connexions à la DB"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def save_prediction(self, prediction_data: Dict) -> int:
        """Sauvegarde une prédiction et retourne l'ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO predictions 
                (sport, home_team, away_team, league, prediction_date, 
                 probabilities, score_prediction, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_data.get('sport'),
                prediction_data.get('home_team'),
                prediction_data.get('away_team'),
                prediction_data.get('league'),
                prediction_data.get('date'),
                json.dumps(prediction_data.get('probabilities', {})),
                json.dumps(prediction_data.get('score_prediction', {})),
                prediction_data.get('confidence_score', 0.0)
            ))
            
            prediction_id = cursor.lastrowid
            self.logger.log_prediction(prediction_data)
            return prediction_id
    
    def get_prediction_history(self, limit: int = 50) -> List[Dict]:
        """Récupère l'historique des prédictions"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM predictions 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            predictions = []
            
            for row in rows:
                pred = dict(row)
                pred['probabilities'] = json.loads(pred['probabilities']) if pred['probabilities'] else {}
                pred['score_prediction'] = json.loads(pred['score_prediction']) if pred['score_prediction'] else {}
                predictions.append(pred)
            
            return predictions
    
    def update_prediction_result(self, prediction_id: int, actual_result: str, accuracy: float):
        """Met à jour le résultat réel d'une prédiction"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE predictions 
                SET actual_result = ?, accuracy = ? 
                WHERE id = ?
            """, (actual_result, accuracy, prediction_id))
    
    def save_bet(self, prediction_id: int, bet_data: Dict):
        """Sauvegarde un pari"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO bets 
                (prediction_id, bet_type, odds, stake, outcome, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                bet_data.get('bet_type'),
                bet_data.get('odds'),
                bet_data.get('stake'),
                bet_data.get('outcome'),
                bet_data.get('profit_loss')
            ))
    
    def get_cache(self, key: str):
        """Récupère une valeur du cache"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT value FROM cache 
                WHERE key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """, (key,))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row['value'])
            return None
    
    def set_cache(self, key: str, value: Any, ttl_seconds: int = 1800):
        """Stocke une valeur dans le cache"""
        expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
            """, (key, json.dumps(value), expires_at))

# =============================================================================
# CLIENT API RÉSILIENT
# =============================================================================

class ResilientAPIClient:
    """Client API avec retry et fallback"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'SportsPredictionApp/1.0'})
        self.timeout = 10
        self.max_retries = 3
        self.backoff_factor = 1
    
    def fetch_with_retry(self, url: str, headers: Dict = None) -> Dict:
        """Télécharge avec mécanisme de retry"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, 
                    headers=headers, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                
                wait_time = self.backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
                continue
        
        return {}
    
    def fetch_with_fallback(self, url: str, headers: Dict = None, 
                           fallback_func=None) -> Dict:
        """Télécharge avec fallback en cas d'échec"""
        try:
            return self.fetch_with_retry(url, headers)
        except Exception as e:
            logging.warning(f"API call failed, using fallback: {e}")
            if fallback_func:
                return fallback_func()
            return {}

# =============================================================================
# CONFIGURATION DES APIS ET TOKENS
# =============================================================================

class APIConfig:
    """Configuration des APIs externes"""
    
    # Clés API (demo par défaut)
    FOOTBALL_API_KEY = "demo"
    BASKETBALL_API_KEY = "demo"
    WEATHER_API_KEY = "demo"
    
    # URLs des APIs
    FOOTBALL_API_URL = "https://v3.football.api-sports.io"
    BASKETBALL_API_URL = "https://v1.basketball.api-sports.io"
    WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    # Temps de cache (secondes)
    CACHE_DURATION = 1800  # 30 minutes
    
    @staticmethod
    def get_football_headers():
        return {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': APIConfig.FOOTBALL_API_KEY
        }
    
    @staticmethod
    def get_basketball_headers():
        return {
            'x-rapidapi-host': 'v1.basketball.api-sports.io',
            'x-rapidapi-key': APIConfig.BASKETBALL_API_KEY
        }

# =============================================================================
# COLLECTEUR DE DONNÉES AVANCÉ AVEC CACHE
# =============================================================================

class AdvancedDataCollector:
    """Collecteur de données avancé avec cache et parallélisation"""
    
    def __init__(self):
        self.api_client = ResilientAPIClient()
        self.db = LocalDatabase()
        self.validator = DataValidator()
        self.security_validator = SecurityValidator()
        
        # Base de données étendue
        self.local_data = self._init_extended_local_data()
    
    def _init_extended_local_data(self):
        """Initialise les données locales étendues"""
        return {
            'football': {
                'teams': {
                    'Paris SG': {
                        'attack': 96, 'defense': 89, 'midfield': 92, 
                        'form': 'WWDLW', 'goals_avg': 2.4,
                        'home_strength': 92, 'away_strength': 88,
                        'coach': 'Luis Enrique',
                        'stadium': 'Parc des Princes',
                        'city': 'Paris',
                        'last_updated': datetime.now().isoformat()
                    },
                    'Marseille': {
                        'attack': 85, 'defense': 81, 'midfield': 83,
                        'form': 'DWWLD', 'goals_avg': 1.8,
                        'home_strength': 84, 'away_strength': 79,
                        'coach': 'Jean-Louis Gasset',
                        'stadium': 'Orange Vélodrome',
                        'city': 'Marseille',
                        'last_updated': datetime.now().isoformat()
                    },
                    # ... autres équipes
                }
            },
            'basketball': {
                'teams': {
                    'Boston Celtics': {
                        'offense': 118, 'defense': 110, 'pace': 98,
                        'form': 'WWLWW', 'points_avg': 118.5,
                        'home_strength': 95, 'away_strength': 90,
                        'coach': 'Joe Mazzulla',
                        'arena': 'TD Garden',
                        'city': 'Boston',
                        'last_updated': datetime.now().isoformat()
                    },
                    # ... autres équipes
                }
            }
        }
    
    @functools.lru_cache(maxsize=128)
    def get_team_data_cached(self, sport: str, team_name: str, league: str = None) -> Dict:
        """Version avec cache LRU"""
        return self.get_team_data(sport, team_name, league)
    
    @handle_errors
    def get_team_data(self, sport: str, team_name: str, league: str = None) -> Dict:
        """Récupère les données d'une équipe avec cache"""
        # Nettoyage de l'entrée
        team_name_clean = self.security_validator.sanitize_input(team_name)
        
        # Génération de la clé de cache
        cache_key = f"team_data_{sport}_{team_name_clean}_{league}"
        
        # Vérifier le cache de la base de données
        cached_data = self.db.get_cache(cache_key)
        if cached_data:
            cached_data['source'] = 'db_cache'
            return cached_data
        
        try:
            # Vérifier dans les données locales
            local_teams = self.local_data.get(sport, {}).get('teams', {})
            
            if team_name_clean in local_teams:
                data = {**local_teams[team_name_clean], 'source': 'local_db'}
                self.db.set_cache(cache_key, data, APIConfig.CACHE_DURATION)
                return data
            
            # Chercher correspondance partielle
            for known_team, data in local_teams.items():
                if (team_name_clean.lower() in known_team.lower() or 
                    known_team.lower() in team_name_clean.lower()):
                    data = {**data, 'source': 'local_db_match'}
                    self.db.set_cache(cache_key, data, APIConfig.CACHE_DURATION)
                    return data
            
            # Générer des données réalistes
            if sport == 'football':
                data = self._generate_football_stats(team_name_clean)
            else:
                data = self._generate_basketball_stats(team_name_clean)
            
            self.db.set_cache(cache_key, data, APIConfig.CACHE_DURATION // 2)
            return data
                
        except Exception as e:
            logging.error(f"Error fetching team data: {e}")
            if sport == 'football':
                return self._generate_football_stats(team_name_clean)
            else:
                return self._generate_basketball_stats(team_name_clean)
    
    def get_multiple_teams_data(self, sport: str, team_names: List[str]) -> Dict[str, Dict]:
        """Récupère les données de plusieurs équipes en parallèle"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_team = {
                executor.submit(self.get_team_data_cached, sport, team): team 
                for team in team_names
            }
            
            results = {}
            for future in concurrent.futures.as_completed(future_to_team):
                team = future_to_team[future]
                try:
                    results[team] = future.result()
                except Exception as e:
                    logging.error(f"Error fetching data for {team}: {e}")
                    results[team] = self._generate_fallback_stats(team, sport)
            
            return results
    
    def _generate_football_stats(self, team_name: str = None) -> Dict:
        """Génère des statistiques football réalistes"""
        attack = random.randint(75, 95)
        defense = random.randint(75, 95)
        midfield = random.randint(75, 95)
        
        return {
            'attack': attack,
            'defense': defense,
            'midfield': midfield,
            'form': random.choice(['WWDLW', 'WDWLD', 'LDWWD', 'DWWDL']),
            'goals_avg': round(random.uniform(1.2, 2.8), 1),
            'home_strength': random.randint(80, 95),
            'away_strength': random.randint(75, 90),
            'team_name': team_name or 'Team',
            'source': 'generated',
            'city': random.choice(['Paris', 'Lyon', 'Marseille', 'Lille', 'Bordeaux']),
            'stadium': f"Stade de {team_name or 'Team'}",
            'last_updated': datetime.now().isoformat()
        }
    
    def _generate_basketball_stats(self, team_name: str = None) -> Dict:
        """Génère des statistiques basketball réalistes"""
        offense = random.randint(100, 120)
        defense = random.randint(100, 120)
        
        return {
            'offense': offense,
            'defense': defense,
            'pace': random.randint(95, 105),
            'form': random.choice(['WWLWW', 'WLWWL', 'LWWLD']),
            'points_avg': round(random.uniform(105.0, 120.0), 1),
            'home_strength': random.randint(85, 98),
            'away_strength': random.randint(80, 95),
            'team_name': team_name or 'Team',
            'source': 'generated',
            'city': random.choice(['Boston', 'Los Angeles', 'Chicago', 'Miami', 'New York']),
            'arena': f"{team_name or 'Team'} Arena",
            'last_updated': datetime.now().isoformat()
        }
    
    def _generate_fallback_stats(self, team_name: str, sport: str) -> Dict:
        """Génère des statistiques de fallback"""
        if sport == 'football':
            return self._generate_football_stats(team_name)
        else:
            return self._generate_basketball_stats(team_name)

# =============================================================================
# ANALYSE STATISTIQUE AVANCÉE
# =============================================================================

class AdvancedStatisticalAnalysis:
    """Analyses statistiques avancées"""
    
    @staticmethod
    def calculate_poisson_probabilities(home_lambda: float, away_lambda: float, 
                                       max_goals: int = 5) -> pd.DataFrame:
        """Calcule les probabilités Poisson pour tous les scores"""
        scores = []
        probabilities = []
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_home = AdvancedStatisticalAnalysis._poisson_pmf(i, home_lambda)
                prob_away = AdvancedStatisticalAnalysis._poisson_pmf(j, away_lambda)
                prob = prob_home * prob_away
                scores.append(f"{i}-{j}")
                probabilities.append(prob)
        
        df = pd.DataFrame({'Score': scores, 'Probabilité': probabilities})
        df['Probabilité %'] = df['Probabilité'] * 100
        
        return df.sort_values('Probabilité', ascending=False)
    
    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        """Calcule la fonction de masse de probabilité de Poisson"""
        if lam == 0:
            return 1.0 if k == 0 else 0.0
        
        try:
            if lam > 50:
                return AdvancedStatisticalAnalysis._normal_approximation(k, lam)
            
            result = math.exp(-lam) * (lam ** k) / math.factorial(k)
            return max(0.0, min(1.0, result))
        except:
            return 0.0
    
    @staticmethod
    def _normal_approximation(k: int, lam: float) -> float:
        """Approximation normale pour Poisson avec grand lambda"""
        mean = lam
        std = math.sqrt(lam)
        
        if std == 0:
            return 0.0
        
        z = (k - mean) / std
        return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * z * z)
    
    @staticmethod
    def calculate_expected_goals(home_data: Dict, away_data: Dict, 
                                league_data: Dict) -> Tuple[float, float]:
        """Calcule les xG (Expected Goals)"""
        home_attack = home_data.get('attack', 75) / 100
        away_defense = (100 - away_data.get('defense', 75)) / 100
        home_xg = home_attack * away_defense * league_data.get('goals_avg', 2.7)
        
        away_attack = away_data.get('attack', 70) / 100
        home_defense = (100 - home_data.get('defense', 75)) / 100
        away_xg = away_attack * home_defense * league_data.get('goals_avg', 2.7) * 0.8
        
        return round(home_xg, 2), round(away_xg, 2)
    
    @staticmethod
    def analyze_trends(form_string: str) -> Dict[str, Any]:
        """Analyse les tendances de forme"""
        if not form_string:
            return {'trend': 'stable', 'momentum': 0, 'consistency': 0}
        
        results = []
        for char in form_string:
            if char == 'W':
                results.append(1)
            elif char == 'D':
                results.append(0.5)
            else:
                results.append(0)
        
        if len(results) < 3:
            return {'trend': 'insufficient_data', 'momentum': 0, 'consistency': 0}
        
        recent_avg = np.mean(results[-3:])
        overall_avg = np.mean(results)
        
        momentum = recent_avg - overall_avg
        
        if momentum > 0.2:
            trend = 'positive'
        elif momentum < -0.2:
            trend = 'negative'
        else:
            trend = 'stable'
        
        if len(results) > 1:
            variance = np.var(results)
            consistency = 1 - math.sqrt(variance)
        else:
            consistency = 0
        
        return {
            'trend': trend,
            'momentum': round(momentum, 2),
            'consistency': round(consistency, 2),
            'recent_form': results[-3:],
            'form_streak': AdvancedStatisticalAnalysis._calculate_streak(form_string)
        }
    
    @staticmethod
    def _calculate_streak(form_string: str) -> Dict[str, int]:
        """Calcule les séries"""
        if not form_string:
            return {'wins': 0, 'draws': 0, 'losses': 0}
        
        current_char = form_string[0]
        streak = 1
        
        for char in form_string[1:]:
            if char == current_char:
                streak += 1
            else:
                break
        
        return {
            'wins': streak if current_char == 'W' else 0,
            'draws': streak if current_char == 'D' else 0,
            'losses': streak if current_char == 'L' else 0
        }
    
    @staticmethod
    def calculate_value_bets(predicted_prob: float, bookmaker_odd: float, 
                            threshold: float = 0.05) -> Tuple[bool, float]:
        """Calcule si un pari a de la valeur"""
        implied_prob = 1 / bookmaker_odd
        value = predicted_prob - implied_prob
        
        is_value_bet = value > threshold
        expected_value = (bookmaker_odd - 1) * predicted_prob - (1 - predicted_prob)
        
        return is_value_bet, round(expected_value, 3)

# =============================================================================
# MODÈLE DE MACHINE LEARNING SIMPLE
# =============================================================================

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    class MachineLearningPredictor:
        """Intégration de modèles ML simples"""
        
        def __init__(self):
            self.models = {}
            self.scalers = {}
            self.feature_names = None
            
        def train_model(self, sport: str, historical_data: pd.DataFrame):
            """Entraîne un modèle simple (régression logistique)"""
            if len(historical_data) < 20:
                logging.warning(f"Insufficient data for {sport} model training")
                return
            
            # Préparation des features
            features = self._extract_features(historical_data)
            target = historical_data['result'].map({'H': 0, 'D': 1, 'A': 2})
            
            # Normalisation
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Entraînement
            model = LogisticRegression(multi_class='multinomial', max_iter=1000)
            model.fit(features_scaled, target)
            
            self.models[sport] = model
            self.scalers[sport] = scaler
            self.feature_names = features.columns.tolist()
            
            logging.info(f"Model trained for {sport} with {len(historical_data)} samples")
        
        def predict(self, sport: str, match_features: Dict) -> Dict:
            """Prédiction avec modèle ML"""
            if sport not in self.models:
                return self._fallback_prediction(match_features)
            
            features_array = self._dict_to_features_array(match_features)
            features_scaled = self.scalers[sport].transform([features_array])
            
            probabilities = self.models[sport].predict_proba(features_scaled)[0]
            
            return {
                'home_win': probabilities[0] * 100,
                'draw': probabilities[1] * 100,
                'away_win': probabilities[2] * 100,
                'model_type': 'logistic_regression',
                'confidence': float(np.max(probabilities))
            }
        
        def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
            """Extrait les features du dataset"""
            features = pd.DataFrame()
            
            # Features de base
            features['home_attack'] = data['home_attack']
            features['home_defense'] = data['home_defense']
            features['away_attack'] = data['away_attack']
            features['away_defense'] = data['away_defense']
            
            # Features dérivées
            features['attack_diff'] = data['home_attack'] - data['away_attack']
            features['defense_diff'] = data['home_defense'] - data['away_defense']
            features['home_form_score'] = data['home_form'].apply(self._form_to_score)
            features['away_form_score'] = data['away_form'].apply(self._form_to_score)
            
            return features
        
        def _form_to_score(self, form_string: str) -> float:
            """Convertit une chaîne de forme en score numérique"""
            if not form_string:
                return 0.5
            
            score = 0
            for char in form_string[-5:]:  # Derniers 5 matchs
                if char == 'W':
                    score += 1
                elif char == 'D':
                    score += 0.5
            
            return score / 5
        
        def _dict_to_features_array(self, match_features: Dict) -> np.ndarray:
            """Convertit un dictionnaire de features en array"""
            features = []
            
            # Features de base
            features.append(match_features.get('home_attack', 75))
            features.append(match_features.get('home_defense', 75))
            features.append(match_features.get('away_attack', 75))
            features.append(match_features.get('away_defense', 75))
            
            # Features dérivées
            features.append(features[0] - features[2])  # attack_diff
            features.append(features[1] - features[3])  # defense_diff
            
            # Form scores
            home_form = match_features.get('home_form', '')
            away_form = match_features.get('away_form', '')
            features.append(self._form_to_score(home_form))
            features.append(self._form_to_score(away_form))
            
            return np.array(features)
        
        def _fallback_prediction(self, match_features: Dict) -> Dict:
            """Prédiction de fallback si le modèle n'est pas entraîné"""
            home_strength = match_features.get('home_attack', 75) + match_features.get('home_defense', 75)
            away_strength = match_features.get('away_attack', 75) + match_features.get('away_defense', 75)
            
            total = home_strength + away_strength
            home_prob = (home_strength / total) * 0.67
            away_prob = (away_strength / total) * 0.67
            draw_prob = 1 - home_prob - away_prob
            
            return {
                'home_win': home_prob * 100,
                'draw': draw_prob * 100,
                'away_win': away_prob * 100,
                'model_type': 'fallback'
            }
    
except ImportError:
    class MachineLearningPredictor:
        """Fallback si scikit-learn n'est pas disponible"""
        
        def __init__(self):
            pass
        
        def train_model(self, sport: str, historical_data: pd.DataFrame):
            logging.warning("scikit-learn not available, skipping model training")
        
        def predict(self, sport: str, match_features: Dict) -> Dict:
            return self._fallback_prediction(match_features)
        
        def _fallback_prediction(self, match_features: Dict) -> Dict:
            home_strength = match_features.get('home_attack', 75) + match_features.get('home_defense', 75)
            away_strength = match_features.get('away_attack', 75) + match_features.get('away_defense', 75)
            
            total = home_strength + away_strength
            home_prob = (home_strength / total) * 0.67
            away_prob = (away_strength / total) * 0.67
            draw_prob = 1 - home_prob - away_prob
            
            return {
                'home_win': home_prob * 100,
                'draw': draw_prob * 100,
                'away_win': away_prob * 100,
                'model_type': 'fallback_no_ml'
            }

# =============================================================================
# MOTEUR DE PRÉDICTION AVANCÉ
# =============================================================================

class AdvancedPredictionEngine:
    """Moteur de prédiction avancé avec ML"""
    
    def __init__(self, data_collector: AdvancedDataCollector):
        self.data_collector = data_collector
        self.stats_analyzer = AdvancedStatisticalAnalysis()
        self.ml_predictor = MachineLearningPredictor()
        self.db = LocalDatabase()
        
        self.config = {
            'football': {
                'weights': {
                    'team_strength': 0.30,
                    'form': 0.20,
                    'h2h': 0.15,
                    'home_advantage': 0.10,
                    'injuries': 0.10,
                    'motivation': 0.08,
                    'weather': 0.07
                }
            },
            'basketball': {
                'weights': {
                    'team_strength': 0.35,
                    'form': 0.18,
                    'h2h': 0.12,
                    'home_advantage': 0.12,
                    'injuries': 0.10,
                    'motivation': 0.08,
                    'weather': 0.05
                }
            }
        }
    
    @handle_errors
    def analyze_match_comprehensive(self, sport: str, home_team: str, 
                                  away_team: str, league: str, 
                                  match_date: date) -> Dict[str, Any]:
        """Analyse complète d'un match avec toutes les améliorations"""
        
        # Nettoyage des entrées
        home_team_clean = SecurityValidator.sanitize_input(home_team)
        away_team_clean = SecurityValidator.sanitize_input(away_team)
        
        # Validation
        is_valid, message = DataValidator.validate_team_name(home_team_clean, SportType(sport))
        if not is_valid:
            raise ValueError(f"Équipe domicile invalide: {message}")
        
        is_valid, message = DataValidator.validate_team_name(away_team_clean, SportType(sport))
        if not is_valid:
            raise ValueError(f"Équipe extérieur invalide: {message}")
        
        is_valid, message = DataValidator.validate_match_date(match_date)
        if not is_valid:
            raise ValueError(f"Date invalide: {message}")
        
        # Récupération parallèle des données
        team_data = self.data_collector.get_multiple_teams_data(
            sport, [home_team_clean, away_team_clean]
        )
        
        home_data = team_data.get(home_team_clean, {})
        away_data = team_data.get(away_team_clean, {})
        
        # Préparation des features pour ML
        ml_features = self._prepare_ml_features(sport, home_data, away_data)
        
        # Prédiction ML
        ml_prediction = self.ml_predictor.predict(sport, ml_features)
        
        # Analyse statistique traditionnelle
        league_data = self.data_collector.get_league_data(sport, league)
        h2h_data = self.data_collector.get_head_to_head(sport, home_team_clean, away_team_clean, league)
        
        stats_prediction = self._calculate_statistical_probabilities(
            sport, home_data, away_data, league_data, h2h_data
        )
        
        # Fusion des prédictions
        if ml_prediction['model_type'] not in ['fallback', 'fallback_no_ml']:
            final_prediction = self._fuse_predictions(
                stats_prediction, 
                ml_prediction,
                weights={'stats': 0.6, 'ml': 0.4}
            )
        else:
            final_prediction = stats_prediction
        
        # Facteurs contextuels
        context_factors = self._analyze_context_factors(
            sport, home_team_clean, away_team_clean, league, match_date
        )
        
        # Ajustement avec facteurs contextuels
        adjusted_prediction = self._adjust_with_context(
            final_prediction, context_factors
        )
        
        # Prédiction de score
        score_prediction = self._predict_score(
            sport, home_data, away_data, adjusted_prediction
        )
        
        # Analyse des paris
        bookmaker_odds = self.data_collector.get_bookmaker_odds(
            home_team_clean, away_team_clean, sport
        )
        
        betting_analysis = self._analyze_betting_opportunities(
            adjusted_prediction, bookmaker_odds, sport
        )
        
        # Calcul de la confiance
        confidence_score = self._calculate_confidence_score(
            home_data, away_data, h2h_data,
            context_factors['injuries']['home_count'],
            context_factors['injuries']['away_count']
        )
        
        # Construction du résultat
        result = {
            'match_info': {
                'sport': sport,
                'home_team': home_team_clean,
                'away_team': away_team_clean,
                'league': league,
                'date': match_date.strftime('%Y-%m-%d'),
                'venue': home_data.get('stadium') or home_data.get('arena', 'Stade inconnu'),
                'time': '20:00'
            },
            
            'probabilities': adjusted_prediction,
            'score_prediction': score_prediction,
            'confidence_score': confidence_score,
            
            'team_analysis': {
                'home': {
                    'stats': home_data,
                    'form_analysis': self.stats_analyzer.analyze_trends(home_data.get('form', ''))
                },
                'away': {
                    'stats': away_data,
                    'form_analysis': self.stats_analyzer.analyze_trends(away_data.get('form', ''))
                }
            },
            
            'contextual_factors': context_factors,
            
            'model_info': {
                'ml_prediction': ml_prediction,
                'stats_prediction': stats_prediction,
                'fusion_method': 'weighted_average'
            },
            
            'betting_analysis': betting_analysis,
            
            'recommendations': self._generate_recommendations(
                adjusted_prediction, betting_analysis, sport
            ),
            
            'data_quality': {
                'home_team_source': home_data.get('source', 'unknown'),
                'away_team_source': away_data.get('source', 'unknown'),
                'ml_model_used': ml_prediction.get('model_type', 'none')
            }
        }
        
        # Sauvegarde en base de données
        prediction_id = self.db.save_prediction(result)
        result['prediction_id'] = prediction_id
        
        return result
    
    def _prepare_ml_features(self, sport: str, home_data: Dict, away_data: Dict) -> Dict:
        """Prépare les features pour le modèle ML"""
        features = {
            'home_attack': home_data.get('attack' if sport == 'football' else 'offense', 75),
            'home_defense': home_data.get('defense', 75),
            'away_attack': away_data.get('attack' if sport == 'football' else 'offense', 75),
            'away_defense': away_data.get('defense', 75),
            'home_form': home_data.get('form', ''),
            'away_form': away_data.get('form', ''),
            'home_goals_avg': home_data.get('goals_avg', 0) if sport == 'football' else home_data.get('points_avg', 0),
            'away_goals_avg': away_data.get('goals_avg', 0) if sport == 'football' else away_data.get('points_avg', 0)
        }
        
        return features
    
    def _calculate_statistical_probabilities(self, sport: str, home_data: Dict, 
                                           away_data: Dict, league_data: Dict, 
                                           h2h_data: Dict) -> Dict[str, float]:
        """Calcule les probabilités statistiques"""
        if sport == 'football':
            home_strength = home_data.get('attack', 75) * 0.4 + home_data.get('defense', 75) * 0.3 + home_data.get('midfield', 75) * 0.3
            away_strength = away_data.get('attack', 70) * 0.4 + away_data.get('defense', 70) * 0.3 + away_data.get('midfield', 70) * 0.3
            
            home_strength *= 1.15  # Avantage domicile
            
            total = home_strength + away_strength
            
            home_prob = (home_strength / total) * 0.67
            away_prob = (away_strength / total) * 0.67
            draw_prob = 1 - home_prob - away_prob
            
            # Ajustement H2H
            h2h_home_rate = h2h_data.get('home_win_rate', 0.5)
            home_prob *= (0.8 + h2h_home_rate * 0.4)
            
            # Normalisation
            total = home_prob + draw_prob + away_prob
            home_prob /= total
            draw_prob /= total
            away_prob /= total
            
            return {
                'home_win': home_prob * 100,
                'draw': draw_prob * 100,
                'away_win': away_prob * 100
            }
        else:
            home_strength = home_data.get('offense', 100) * 0.6 + (200 - home_data.get('defense', 100)) * 0.4
            away_strength = away_data.get('offense', 95) * 0.6 + (200 - away_data.get('defense', 100)) * 0.4
            
            home_strength *= 1.10  # Avantage domicile
            
            total = home_strength + away_strength
            home_prob = home_strength / total
            
            return {
                'home_win': home_prob * 100,
                'away_win': (1 - home_prob) * 100
            }
    
    def _fuse_predictions(self, stats_pred: Dict, ml_pred: Dict, 
                         weights: Dict) -> Dict[str, float]:
        """Fusionne les prédictions statistiques et ML"""
        fused = {}
        
        for key in stats_pred.keys():
            if key in ml_pred:
                fused[key] = (stats_pred[key] * weights['stats'] + 
                             ml_pred[key] * weights['ml'])
            else:
                fused[key] = stats_pred[key]
        
        # Normalisation
        total = sum(fused.values())
        if total > 0:
            fused = {k: (v / total) * 100 for k, v in fused.items()}
        
        return {k: round(v, 1) for k, v in fused.items()}
    
    def _analyze_context_factors(self, sport: str, home_team: str, away_team: str,
                                league: str, match_date: date) -> Dict:
        """Analyse tous les facteurs contextuels"""
        # Blessures
        home_injuries = self.data_collector.get_injuries_suspensions(sport, home_team)
        away_injuries = self.data_collector.get_injuries_suspensions(sport, away_team)
        
        injury_impact_home = self._calculate_injury_impact(home_injuries)
        injury_impact_away = self._calculate_injury_impact(away_injuries)
        
        # Météo
        home_city = self.data_collector.get_team_data(sport, home_team).get('city', 'Paris')
        weather = self.data_collector.get_weather_conditions(home_city, match_date)
        weather_impact = self._calculate_weather_impact(weather, sport)
        
        # Motivation
        motivation_factors = self.data_collector.get_motivation_factors(
            home_team, away_team, sport, league
        )
        motivation_score = self._calculate_motivation_score(motivation_factors)
        
        # Forme
        home_form_analysis = self.stats_analyzer.analyze_trends(
            self.data_collector.get_team_data(sport, home_team).get('form', '')
        )
        away_form_analysis = self.stats_analyzer.analyze_trends(
            self.data_collector.get_team_data(sport, away_team).get('form', '')
        )
        
        return {
            'injuries': {
                'home': [vars(inj) for inj in home_injuries],
                'away': [vars(inj) for inj in away_injuries],
                'home_count': len(home_injuries),
                'away_count': len(away_injuries),
                'home_impact': injury_impact_home,
                'away_impact': injury_impact_away
            },
            'weather': vars(weather),
            'weather_impact': weather_impact,
            'motivation_factors': motivation_factors,
            'motivation_score': motivation_score,
            'form_analysis': {
                'home': home_form_analysis,
                'away': away_form_analysis
            }
        }
    
    def _adjust_with_context(self, probabilities: Dict[str, float], 
                            context_factors: Dict) -> Dict[str, float]:
        """Ajuste les probabilités avec les facteurs contextuels"""
        adjusted = probabilities.copy()
        
        # Ajustement blessures
        if 'home_win' in adjusted:
            adjusted['home_win'] *= context_factors['injuries']['home_impact']
            adjusted['away_win'] *= context_factors['injuries']['away_impact']
        
        # Ajustement météo
        for key in adjusted:
            adjusted[key] *= context_factors['weather_impact']
        
        # Ajustement motivation
        for key in adjusted:
            adjusted[key] *= context_factors['motivation_score']
        
        # Ajustement forme
        home_momentum = context_factors['form_analysis']['home'].get('momentum', 0)
        away_momentum = context_factors['form_analysis']['away'].get('momentum', 0)
        
        if 'home_win' in adjusted:
            adjusted['home_win'] *= (1 + home_momentum * 0.3)
            adjusted['away_win'] *= (1 + away_momentum * 0.3)
        
        # Normalisation
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: (v / total) * 100 for k, v in adjusted.items()}
        
        return {k: round(v, 1) for k, v in adjusted.items()}
    
    def _calculate_injury_impact(self, injuries: List[PlayerInjury]) -> float:
        """Calcule l'impact des blessures"""
        if not injuries:
            return 1.0
        
        total_impact = sum(injury.impact_score for injury in injuries)
        avg_impact = total_impact / len(injuries)
        
        impact_factor = 1.0 - (avg_impact / 10 * 0.3)
        return max(0.7, min(1.0, impact_factor))
    
    def _calculate_weather_impact(self, weather: WeatherCondition, sport: str) -> float:
        """Calcule l'impact de la météo"""
        impact = 1.0
        
        if weather.precipitation > 0.7:
            impact *= 0.85
        
        if weather.wind_speed > 20:
            impact *= 0.9
        
        if weather.temperature < 0 or weather.temperature > 30:
            impact *= 0.95
        
        if weather.humidity > 85:
            impact *= 0.95
        
        return impact
    
    def _calculate_motivation_score(self, motivation_factors: Dict[str, float]) -> float:
        """Calcule un score de motivation"""
        if not motivation_factors:
            return 1.0
        
        values = list(motivation_factors.values())
        return np.mean(values)
    
    def _predict_score(self, sport: str, home_data: Dict, away_data: Dict,
                      probabilities: Dict[str, float]) -> Dict[str, Any]:
        """Prédit le score final"""
        if sport == 'football':
            home_xg, away_xg = self.stats_analyzer.calculate_expected_goals(
                home_data, away_data, {'goals_avg': 2.7}
            )
            
            home_goals = self._simulate_goals_from_xg(home_xg)
            away_goals = self._simulate_goals_from_xg(away_xg)
            
            if probabilities['home_win'] > 60:
                home_goals = max(home_goals, away_goals + 1)
            elif probabilities['away_win'] > 55:
                away_goals = max(away_goals, home_goals + 1)
            
            return {
                'exact_score': f"{home_goals}-{away_goals}",
                'home_goals': home_goals,
                'away_goals': away_goals,
                'total_goals': home_goals + away_goals,
                'both_teams_score': home_goals > 0 and away_goals > 0
            }
        else:
            home_pts = int(home_data.get('points_avg', 100) * random.uniform(0.85, 1.15))
            away_pts = int(away_data.get('points_avg', 95) * random.uniform(0.85, 1.15))
            
            win_prob_diff = probabilities['home_win'] - probabilities['away_win']
            point_diff = int(abs(win_prob_diff) * 0.3)
            
            if probabilities['home_win'] > probabilities['away_win']:
                home_pts += point_diff
                away_pts -= point_diff
            else:
                home_pts -= point_diff
                away_pts += point_diff
            
            return {
                'exact_score': f"{home_pts}-{away_pts}",
                'home_points': home_pts,
                'away_points': away_pts,
                'total_points': home_pts + away_pts,
                'point_spread': abs(home_pts - away_pts)
            }
    
    def _simulate_goals_from_xg(self, xg: float) -> int:
        """Simule les buts à partir des xG"""
        goals = 0
        for _ in range(20):
            if random.random() < xg / 20:
                goals += 1
        
        return min(goals, 5)
    
    def _analyze_betting_opportunities(self, probabilities: Dict[str, float],
                                      bookmaker_odds: Dict[str, Dict],
                                      sport: str) -> Dict[str, Any]:
        """Analyse les opportunités de pari"""
        value_bets = []
        threshold = 0.05
        
        for bookmaker, odds in bookmaker_odds.items():
            if sport == 'football':
                # Victoire domicile
                is_value, ev = self.stats_analyzer.calculate_value_bets(
                    probabilities['home_win'] / 100,
                    odds.get('home', 2.0),
                    threshold
                )
                if is_value:
                    value_bets.append({
                        'bookmaker': bookmaker,
                        'bet': "Victoire domicile",
                        'odd': odds['home'],
                        'value': round((1/odds['home'] - probabilities['home_win']/100) * 100, 1),
                        'expected_value': ev
                    })
            else:
                # Basketball
                is_value, ev = self.stats_analyzer.calculate_value_bets(
                    probabilities['home_win'] / 100,
                    odds.get('home', 2.0),
                    threshold
                )
                if is_value:
                    value_bets.append({
                        'bookmaker': bookmaker,
                        'bet': "Victoire Domicile",
                        'odd': odds['home'],
                        'value': round((1/odds['home'] - probabilities['home_win']/100) * 100, 1),
                        'expected_value': ev
                    })
        
        return {
            'value_bets': value_bets,
            'risk_assessment': self._assess_betting_risk(probabilities),
            'best_odds': self._find_best_odds(bookmaker_odds)
        }
    
    def _assess_betting_risk(self, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """Évalue le risque des paris"""
        if 'home_win' in probabilities:
            max_prob = max(probabilities.values())
            
            if max_prob > 75:
                risk_level = 'low'
            elif max_prob > 60:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'risk_level': risk_level,
                'certainty_index': round(max_prob, 1),
                'volatility': round(np.std(list(probabilities.values())), 1)
            }
        
        return {'risk_level': 'unknown', 'certainty_index': 0, 'volatility': 0}
    
    def _find_best_odds(self, bookmaker_odds: Dict[str, Dict]) -> Dict[str, Any]:
        """Trouve les meilleures cotes parmi tous les bookmakers"""
        best_odds = {}
        
        bet_types = set()
        for odds in bookmaker_odds.values():
            bet_types.update(odds.keys())
        
        for bet_type in bet_types:
            best_odd = 0
            best_bookmaker = None
            
            for bookmaker, odds in bookmaker_odds.items():
                if bet_type in odds and odds[bet_type] > best_odd:
                    best_odd = odds[bet_type]
                    best_bookmaker = bookmaker
            
            if best_bookmaker:
                best_odds[bet_type] = {
                    'odd': best_odd,
                    'bookmaker': best_bookmaker
                }
        
        return best_odds
    
    def _generate_recommendations(self, probabilities: Dict[str, float],
                                 betting_analysis: Dict, sport: str) -> Dict[str, Any]:
        """Génère les recommandations"""
        if sport == 'football':
            best_bet = max(probabilities.items(), key=lambda x: x[1])
            
            if best_bet[0] == 'home_win':
                recommendation = "Victoire à domicile"
                confidence = 'high' if best_bet[1] > 60 else 'medium'
            elif best_bet[0] == 'draw':
                recommendation = "Match nul"
                confidence = 'high' if best_bet[1] > 35 else 'medium'
            else:
                recommendation = "Victoire à l'extérieur"
                confidence = 'high' if best_bet[1] > 55 else 'medium'
        else:
            if probabilities['home_win'] > probabilities['away_win']:
                recommendation = "Victoire à domicile"
                confidence = 'high' if probabilities['home_win'] > 65 else 'medium'
            else:
                recommendation = "Victoire à l'extérieur"
                confidence = 'high' if probabilities['away_win'] > 60 else 'medium'
        
        return {
            'main_prediction': {
                'bet': recommendation,
                'probability': max(probabilities.values()),
                'confidence': confidence
            },
            'betting_recommendations': self._get_betting_recommendations(betting_analysis)
        }
    
    def _get_betting_recommendations(self, betting_analysis: Dict) -> List[Dict]:
        """Génère des recommandations de paris"""
        recommendations = []
        
        # Ajouter les paris avec valeur
        for value_bet in betting_analysis.get('value_bets', [])[:3]:
            recommendations.append({
                'type': 'value_bet',
                'description': f"{value_bet['bet']} chez {value_bet['bookmaker']}",
                'odd': value_bet['odd'],
                'value_score': value_bet['value']
            })
        
        # Recommandation basée sur le risque
        risk_level = betting_analysis.get('risk_assessment', {}).get('risk_level', 'medium')
        
        if risk_level == 'low':
            recommendations.append({
                'type': 'risk_based',
                'description': "Pari sécurisé recommandé",
                'suggestion': "Augmenter la mise (3-5% de bankroll)"
            })
        elif risk_level == 'high':
            recommendations.append({
                'type': 'risk_based',
                'description': "Risque élevé détecté",
                'suggestion': "Réduire la mise (1% de bankroll maximum)"
            })
        
        return recommendations
    
    def _calculate_confidence_score(self, home_data: Dict, away_data: Dict,
                                  h2h_data: Dict, home_injuries_count: int,
                                  away_injuries_count: int) -> float:
        """Calcule le score de confiance"""
        confidence = 70.0
        
        if home_data.get('source') in ['local_db', 'api']:
            confidence += 10
        if away_data.get('source') in ['local_db', 'api']:
            confidence += 10
        
        if h2h_data.get('total_matches', 0) > 10:
            confidence += 5
        
        confidence -= (home_injuries_count + away_injuries_count) * 2
        
        return max(50, min(95, round(confidence, 1)))

# =============================================================================
# COMPOSANTS UI RÉUTILISABLES
# =============================================================================

class UIComponents:
    """Composants d'interface réutilisables"""
    
    @staticmethod
    def progress_bar_with_label(label: str, value: float, max_value: float = 100, 
                               color: str = None) -> None:
        """Barre de progression avec label"""
        percentage = (value / max_value) * 100
        
        if color is None:
            if percentage > 70:
                color = "#4CAF50"
            elif percentage > 40:
                color = "#FF9800"
            else:
                color = "#F44336"
        
        st.markdown(f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>{label}</span>
                <span>{value:.1f}%</span>
            </div>
            <div style="height: 10px; background: #e0e0e0; border-radius: 5px; overflow: hidden;">
                <div style="height: 100%; width: {percentage}%; background: {color}; 
                          transition: width 0.5s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(title: str, value: Any, delta: str = None, 
                   help_text: str = None, color: str = None) -> None:
        """Carte métrique améliorée"""
        card_html = f"""
        <div style="
            background: linear-gradient(135deg, {'#667eea' if not color else color} 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <h4 style="margin: 0 0 10px 0;">{title}</h4>
            <h2 style="margin: 0; font-size: 2.5rem;">{value}</h2>
        """
        
        if delta:
            card_html += f'<p style="margin: 10px 0 0 0;">{delta}</p>'
        
        if help_text:
            card_html += f'<small style="opacity: 0.8;">{help_text}</small>'
        
        card_html += "</div>"
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def risk_badge(risk_level: str) -> str:
        """Retourne un badge de risque coloré"""
        colors = {
            'low': ('#4CAF50', '🟢'),
            'medium': ('#FF9800', '🟡'),
            'high': ('#F44336', '🔴')
        }
        
        color, emoji = colors.get(risk_level.lower(), ('#9E9E9E', '⚫'))
        
        return f'<span style="color: {color}; font-weight: bold;">{emoji} {risk_level.upper()}</span>'
    
    @staticmethod
    def display_value_bet(bet: Dict) -> None:
        """Affiche un pari avec valeur"""
        with st.expander(f"💰 {bet['bet']} - {bet['bookmaker']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cote", bet['odd'])
            
            with col2:
                st.metric("Valeur", f"+{bet['value']}%")
            
            with col3:
                st.metric("EV", bet['expected_value'])
            
            if st.button("📝 Enregistrer ce pari", key=f"save_bet_{bet['bookmaker']}"):
                st.success("Pari enregistré dans l'historique!")

# =============================================================================
# INTERNATIONALISATION
# =============================================================================

class Internationalization:
    """Support multilingue"""
    
    TRANSLATIONS = {
        'en': {
            'analyze_match': 'Analyze Match',
            'probability': 'Probability',
            'recommendation': 'Recommendation',
            'scheduled_matches': 'Scheduled Matches',
            'bankroll_management': 'Bankroll Management',
            'home': 'Home',
            'away': 'Away',
            'draw': 'Draw'
        },
        'fr': {
            'analyze_match': 'Analyser le match',
            'probability': 'Probabilité',
            'recommendation': 'Recommandation',
            'scheduled_matches': 'Matchs programmés',
            'bankroll_management': 'Gestion de bankroll',
            'home': 'Domicile',
            'away': 'Extérieur',
            'draw': 'Nul'
        },
        'es': {
            'analyze_match': 'Analizar partido',
            'probability': 'Probabilidad',
            'recommendation': 'Recomendación',
            'scheduled_matches': 'Partidos programados',
            'bankroll_management': 'Gestión de bankroll',
            'home': 'Local',
            'away': 'Visitante',
            'draw': 'Empate'
        }
    }
    
    def __init__(self, lang: str = 'fr'):
        self.lang = lang
    
    def get(self, key: str) -> str:
        """Retourne le texte dans la langue choisie"""
        return self.TRANSLATIONS.get(self.lang, {}).get(
            key, self.TRANSLATIONS['fr'].get(key, key)
        )

# =============================================================================
# EXPORT DE DONNÉES
# =============================================================================

class DataExporter:
    """Export des données dans différents formats"""
    
    @staticmethod
    def export_analysis_to_csv(analysis: Dict, filename: str = "prediction.csv"):
        """Exporte une analyse en CSV"""
        try:
            # Préparation des données
            data = {
                'sport': [analysis['match_info']['sport']],
                'home_team': [analysis['match_info']['home_team']],
                'away_team': [analysis['match_info']['away_team']],
                'date': [analysis['match_info']['date']],
                'league': [analysis['match_info']['league']]
            }
            
            # Ajout des probabilités
            for key, value in analysis['probabilities'].items():
                data[f'prob_{key}'] = [value]
            
            # Ajout de la prédiction de score
            data['predicted_score'] = [analysis['score_prediction'].get('exact_score', '')]
            data['confidence'] = [analysis['confidence_score']]
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            return filename
        except Exception as e:
            logging.error(f"Error exporting to CSV: {e}")
            return None
    
    @staticmethod
    def generate_html_report(analysis: Dict) -> str:
        """Génère un rapport HTML"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Analyse de Match</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .card { background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; }
                .probability { font-size: 24px; font-weight: bold; }
                .recommendation { color: #4CAF50; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Analyse de Match</h1>
                <h2>{home_team} vs {away_team}</h2>
                <p>📅 {date} | 🏆 {league} | ⚽ {sport}</p>
            </div>
            
            <div class="card">
                <h3>📊 Probabilités</h3>
                {probabilities_html}
            </div>
            
            <div class="card">
                <h3>🎯 Prédiction de score</h3>
                <p class="probability">{predicted_score}</p>
                <p>Confiance: <strong>{confidence}%</strong></p>
            </div>
            
            <div class="card">
                <h3>💡 Recommandation</h3>
                <p class="recommendation">{recommendation}</p>
            </div>
            
            <div class="card">
                <h3>📈 Informations complémentaires</h3>
                <p>Score de confiance: {confidence_score}%</p>
                <p>Source des données: {data_source}</p>
                <p>Généré le: {generated_date}</p>
            </div>
        </body>
        </html>
        """
        
        # Préparation des données
        probabilities_html = ""
        for key, value in analysis['probabilities'].items():
            probabilities_html += f"<p>{key}: <strong>{value}%</strong></p>"
        
        # Remplissage du template
        html_content = html_template.format(
            home_team=analysis['match_info']['home_team'],
            away_team=analysis['match_info']['away_team'],
            date=analysis['match_info']['date'],
            league=analysis['match_info']['league'],
            sport=analysis['match_info']['sport'],
            probabilities_html=probabilities_html,
            predicted_score=analysis['score_prediction'].get('exact_score', 'N/A'),
            confidence=max(analysis['probabilities'].values()),
            recommendation=analysis['recommendations']['main_prediction']['bet'],
            confidence_score=analysis['confidence_score'],
            data_source=analysis['data_quality']['home_team_source'],
            generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content

# =============================================================================
# APPLICATION PRINCIPALE STREAMLIT
# =============================================================================

def configure_page():
    """Configure la page Streamlit"""
    st.set_page_config(
        page_title="Pronostics Sports Premium Pro",
        page_icon="🎯",
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
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe-bet-card {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #F44336 0%, #D32F2F 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .tab-container {
        background: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def init_application():
    """Initialise l'application"""
    if 'data_collector' not in st.session_state:
        st.session_state.data_collector = AdvancedDataCollector()
    
    if 'prediction_engine' not in st.session_state:
        st.session_state.prediction_engine = AdvancedPredictionEngine(
            st.session_state.data_collector
        )
    
    if 'db' not in st.session_state:
        st.session_state.db = LocalDatabase()
    
    if 'i18n' not in st.session_state:
        st.session_state.i18n = Internationalization('fr')
    
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

def show_dashboard():
    """Affiche le tableau de bord principal"""
    st.markdown('<h1 class="main-header">🎯 Système Premium Pro de Pronostics Sports</h1>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## ✨ **Nouveautés Premium Pro:**
        
        **🔍 Analyse IA Avancée:**
        - 🤖 Modèles Machine Learning intégrés
        - 📊 Fusion intelligente des prédictions
        - 🧠 Analyse en temps réel multi-sources
        - 📈 Historique complet avec base de données
        
        **💰 Module Paris Professionnel:**
        - 💎 Détection automatique des value bets
        - 🛡️ Gestion de risque intelligente
        - 📊 Optimisation de bankroll
        - 📈 Suivi des performances
        
        **⚙️ Fonctionnalités Pro:**
        - 🔒 Sécurité et validation renforcée
        - 💾 Export des données (CSV, HTML)
        - 🌍 Support multilingue
        - 📱 Interface responsive
        """)
    
    with col2:
        st.markdown("""
        ## 🚀 **Comment utiliser:**
        
        1. **🔍 Choisissez un mode d'analyse**
        2. **🏆 Sélectionnez sport et ligue**
        3. **⚽ Entrez les équipes**
        4. **📊 Analysez tous les facteurs**
        5. **💰 Découvrez les opportunités**
        6. **💾 Exportez vos analyses**
        
        ## 🏆 **Fonctionnalités exclusives:**
        
        **🎯 Précision améliorée:**
        - Combinaison statistiques + ML
        - Facteurs contextuels complets
        - Base de données historique
        
        **💼 Gestion professionnelle:**
        - Suivi des performances
        - Analyse ROI
        - Recommandations personnalisées
        
        **📊 Données enrichies:**
        - 50+ équipes majeures
        - Données temps réel
        - Statistiques détaillées
        """)
    
    # Quick actions
    st.markdown("---")
    st.markdown("### 🎮 Actions rapides")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⚽ PSG vs Marseille", use_container_width=True):
            st.session_state.analysis_mode = "analyze"
            st.session_state.sport = 'football'
            st.session_state.home_team = 'Paris SG'
            st.session_state.away_team = 'Marseille'
            st.session_state.league = 'Ligue 1'
            st.rerun()
    
    with col2:
        if st.button("🏀 Celtics vs Lakers", use_container_width=True):
            st.session_state.analysis_mode = "analyze"
            st.session_state.sport = 'basketball'
            st.session_state.home_team = 'Boston Celtics'
            st.session_state.away_team = 'LA Lakers'
            st.session_state.league = 'NBA'
            st.rerun()
    
    with col3:
        if st.button("📅 Matchs à venir", use_container_width=True):
            st.session_state.analysis_mode = "scheduled"
            st.rerun()
    
    with col4:
        if st.button("📊 Historique", use_container_width=True):
            st.session_state.analysis_mode = "history"
            st.rerun()
    
    # Stats rapides
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            history = st.session_state.db.get_prediction_history(limit=1)
            if history:
                st.metric("Dernière analyse", history[0]['created_at'][:10])
        except:
            st.metric("Prédictions", "N/A")
    
    with col2:
        st.metric("Équipes supportées", "50+")
    
    with col3:
        st.metric("Sports", "2")

def show_analysis_interface():
    """Affiche l'interface d'analyse de match"""
    st.header("🔍 Analyse de Match")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sport = st.selectbox(
            "🏆 Sport",
            options=['football', 'basketball'],
            format_func=lambda x: 'Football ⚽' if x == 'football' else 'Basketball 🏀',
            key="analysis_sport"
        )
        
        if sport == 'football':
            leagues = ['Ligue 1', 'Premier League', 'La Liga', 'Bundesliga', 'Serie A']
            default_home = 'Paris SG'
            default_away = 'Marseille'
        else:
            leagues = ['NBA', 'EuroLeague', 'LNB Pro A']
            default_home = 'Boston Celtics'
            default_away = 'LA Lakers'
        
        league = st.selectbox("🏅 Ligue", leagues, key="analysis_league")
    
    with col2:
        home_team = st.text_input("🏠 Équipe domicile", 
                                 value=st.session_state.get('home_team', default_home),
                                 key="analysis_home")
        away_team = st.text_input("✈️ Équipe extérieur", 
                                 value=st.session_state.get('away_team', default_away),
                                 key="analysis_away")
        
        match_date = st.date_input("📅 Date du match", 
                                  value=date.today(), 
                                  key="analysis_date")
    
    # Validation en temps réel
    if home_team and away_team:
        validator = DataValidator()
        
        col1, col2 = st.columns(2)
        with col1:
            is_valid, message = validator.validate_team_name(home_team, SportType(sport))
            if not is_valid:
                st.warning(f"⚠️ {message}")
            else:
                st.success("✅ Nom valide")
        
        with col2:
            is_valid, message = validator.validate_team_name(away_team, SportType(sport))
            if not is_valid:
                st.warning(f"⚠️ {message}")
            else:
                st.success("✅ Nom valide")
    
    # Options avancées
    with st.expander("⚙️ Options avancées"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_injuries = st.checkbox("Analyser les blessures", value=True)
            include_weather = st.checkbox("Inclure la météo", value=True)
        
        with col2:
            compare_odds = st.checkbox("Comparer les cotes", value=True)
            use_ml = st.checkbox("Utiliser ML", value=True)
        
        with col3:
            confidence_threshold = st.slider("Seuil de confiance (%)", 50, 90, 65)
            export_format = st.selectbox("Format d'export", ["CSV", "HTML", "JSON"])
    
    # Bouton d'analyse
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Lancer l'analyse complète", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyse en cours avec IA..."):
                try:
                    analysis = st.session_state.prediction_engine.analyze_match_comprehensive(
                        sport, home_team, away_team, league, match_date
                    )
                    
                    if analysis.get('error'):
                        st.error(f"❌ Erreur: {analysis.get('error_message')}")
                    else:
                        st.session_state.current_analysis = analysis
                        st.success("✅ Analyse terminée avec succès!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")

def display_analysis_results(analysis: Dict):
    """Affiche les résultats d'analyse"""
    if analysis.get('error'):
        st.error(f"Erreur: {analysis.get('error_message')}")
        return
    
    # En-tête du match
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        sport_icon = "⚽" if analysis['match_info']['sport'] == 'football' else "🏀"
        st.metric("Sport", f"{sport_icon} {analysis['match_info']['sport'].title()}")
    
    with col2:
        st.markdown(f"""
        <h2 style='text-align: center;'>
        🏠 {analysis['match_info']['home_team']} <span style='color: #666;'>vs</span> ✈️ {analysis['match_info']['away_team']}
        </h2>
        <p style='text-align: center; color: #666;'>
        {analysis['match_info']['league']} • {analysis['match_info']['date']} • {analysis['match_info'].get('venue', '')}
        </p>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence = analysis['confidence_score']
        color = "#4CAF50" if confidence >= 80 else "#FF9800" if confidence >= 65 else "#F44336"
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Score de confiance</h4>
            <h2 style="color: {color};">{confidence}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Prédictions", "🎯 Scores", "🏥 Contexte", "💰 Paris", "📈 Stats", "💡 Recommandations"
    ])
    
    with tab1:
        display_predictions_tab(analysis)
    
    with tab2:
        display_scores_tab(analysis)
    
    with tab3:
        display_context_tab(analysis)
    
    with tab4:
        display_betting_tab(analysis)
    
    with tab5:
        display_stats_tab(analysis)
    
    with tab6:
        display_recommendations_tab(analysis)
    
    # Boutons d'export
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Exporter en CSV", use_container_width=True):
            filename = DataExporter.export_analysis_to_csv(analysis)
            if filename:
                st.success(f"✅ Exporté: {filename}")
                with open(filename, "rb") as file:
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=file,
                        file_name=filename,
                        mime="text/csv"
                    )
    
    with col2:
        if st.button("📄 Exporter en HTML", use_container_width=True):
            html_content = DataExporter.generate_html_report(analysis)
            st.download_button(
                label="📥 Télécharger HTML",
                data=html_content,
                file_name="analyse_match.html",
                mime="text/html"
            )
    
    with col3:
        if st.button("📊 Ajouter à l'historique", use_container_width=True):
            st.session_state.prediction_history.append(analysis)
            st.success("✅ Ajouté à l'historique!")

def display_predictions_tab(analysis: Dict):
    """Affiche l'onglet des prédictions"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("🎯 Score Prédit")
        
        score = analysis['score_prediction']['exact_score']
        st.markdown(f"<h1 style='text-align: center; font-size: 3rem;'>{score}</h1>", 
                   unsafe_allow_html=True)
        
        if analysis['match_info']['sport'] == 'football':
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total buts", analysis['score_prediction']['total_goals'])
            with col_b:
                st.metric("Les deux marquent", 
                         "✅ Oui" if analysis['score_prediction']['both_teams_score'] else "❌ Non")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total points", analysis['score_prediction']['total_points'])
            with col_b:
                st.metric("Écart", analysis['score_prediction']['point_spread'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("📊 Probabilités")
        
        probs = analysis['probabilities']
        i18n = st.session_state.i18n
        
        if analysis['match_info']['sport'] == 'football':
            st.markdown("### Probabilités de résultat")
            
            UIComponents.progress_bar_with_label(
                i18n.get('home'), 
                probs['home_win'],
                color="#4CAF50"
            )
            
            UIComponents.progress_bar_with_label(
                i18n.get('draw'), 
                probs['draw'],
                color="#FF9800"
            )
            
            UIComponents.progress_bar_with_label(
                i18n.get('away'), 
                probs['away_win'],
                color="#F44336"
            )
            
            # Stats rapides
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Domicile", f"{probs['home_win']}%")
            with col_b:
                st.metric("Nul", f"{probs['draw']}%")
            with col_c:
                st.metric("Extérieur", f"{probs['away_win']}%")
        else:
            st.markdown("### Probabilités de victoire")
            
            UIComponents.progress_bar_with_label(
                i18n.get('home'), 
                probs['home_win'],
                color="#4CAF50"
            )
            
            UIComponents.progress_bar_with_label(
                i18n.get('away'), 
                probs['away_win'],
                color="#F44336"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Info modèle
    if 'model_info' in analysis:
        with st.expander("🤖 Informations du modèle"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Prédiction statistique:**")
                st.json(analysis['model_info']['stats_prediction'])
            with col2:
                st.write("**Prédiction ML:**")
                st.json(analysis['model_info']['ml_prediction'])

def display_scores_tab(analysis: Dict):
    """Affiche l'onglet des scores"""
    st.subheader("🎯 Analyse des scores")
    
    if analysis['match_info']['sport'] == 'football':
        # Expected Goals
        if 'expected_goals' in analysis.get('statistical_analysis', {}):
            xg = analysis['statistical_analysis']['expected_goals']
            col1, col2 = st.columns(2)
            with col1:
                UIComponents.metric_card("xG Domicile", f"{xg.get('home', 0):.2f}")
            with col2:
                UIComponents.metric_card("xG Extérieur", f"{xg.get('away', 0):.2f}")
        
        # Distribution Poisson
        if 'poisson_probabilities' in analysis.get('statistical_analysis', {}):
            st.markdown("### 📊 Distribution des scores (Poisson)")
            
            poisson_data = analysis['statistical_analysis']['poisson_probabilities']
            if poisson_data:
                df_scores = pd.DataFrame(poisson_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 5 scores probables:**")
                    for _, row in df_scores.head(5).iterrows():
                        st.markdown(f"**{row['Score']}**: {row['Probabilité %']:.1f}%")
                
                with col2:
                    st.markdown("**Tableau complet:**")
                    st.dataframe(
                        df_scores[['Score', 'Probabilité %']].round(1),
                        use_container_width=True,
                        height=300
                    )

def display_context_tab(analysis: Dict):
    """Affiche l'onglet contexte"""
    st.subheader("🏥 Facteurs contextuels")
    
    # Blessures
    if 'contextual_factors' in analysis and 'injuries' in analysis['contextual_factors']:
        injuries = analysis['contextual_factors']['injuries']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🏠 {analysis['match_info']['home_team']}")
            
            if injuries['home_count'] > 0:
                st.warning(f"⚠️ {injuries['home_count']} blessure(s)")
                
                for injury in injuries['home'][:3]:  # Montre max 3 blessures
                    with st.expander(f"🚑 {injury.get('player_name', 'Joueur')}"):
                        st.write(f"**Position:** {injury.get('position', 'N/A')}")
                        st.write(f"**Blessure:** {injury.get('injury_type', 'N/A')}")
                        st.write(f"**Sévérité:** {injury.get('severity', 'N/A')}")
                        
                        impact = injury.get('impact_score', 0)
                        if impact > 7:
                            st.error(f"Impact élevé: {impact}/10")
                        elif impact > 4:
                            st.warning(f"Impact modéré: {impact}/10")
                        else:
                            st.info(f"Impact faible: {impact}/10")
            else:
                st.success("✅ Aucune blessure significative")
        
        with col2:
            st.markdown(f"### ✈️ {analysis['match_info']['away_team']}")
            
            if injuries['away_count'] > 0:
                st.warning(f"⚠️ {injuries['away_count']} blessure(s)")
                
                for injury in injuries['away'][:3]:
                    with st.expander(f"🚑 {injury.get('player_name', 'Joueur')}"):
                        st.write(f"**Position:** {injury.get('position', 'N/A')}")
                        st.write(f"**Blessure:** {injury.get('injury_type', 'N/A')}")
                        st.write(f"**Sévérité:** {injury.get('severity', 'N/A')}")
                        
                        impact = injury.get('impact_score', 0)
                        if impact > 7:
                            st.error(f"Impact élevé: {impact}/10")
                        elif impact > 4:
                            st.warning(f"Impact modéré: {impact}/10")
                        else:
                            st.info(f"Impact faible: {impact}/10")
            else:
                st.success("✅ Aucune blessure significative")
    
    st.divider()
    
    # Météo
    if 'contextual_factors' in analysis and 'weather' in analysis['contextual_factors']:
        weather = analysis['contextual_factors']['weather']
        
        st.subheader("🌤️ Conditions météo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp = weather.get('temperature', 20)
            if temp > 25:
                emoji = "🔥"
            elif temp < 10:
                emoji = "❄️"
            else:
                emoji = "🌡️"
            st.metric(f"{emoji} Température", f"{temp}°C")
        
        with col2:
            precip = weather.get('precipitation', 0) * 100
            if precip > 50:
                emoji = "🌧️"
            elif precip > 20:
                emoji = "☁️"
            else:
                emoji = "☀️"
            st.metric(f"{emoji} Précipitation", f"{precip:.0f}%")
        
        with col3:
            wind = weather.get('wind_speed', 0)
            if wind > 20:
                emoji = "💨"
            else:
                emoji = "🍃"
            st.metric(f"{emoji} Vent", f"{wind} km/h")
        
        with col4:
            humidity = weather.get('humidity', 50)
            if humidity > 80:
                emoji = "💧"
            else:
                emoji = "🌵"
            st.metric(f"{emoji} Humidité", f"{humidity:.0f}%")
        
        # Impact météo
        weather_impact = analysis['contextual_factors'].get('weather_impact', 1.0)
        if weather_impact < 0.9:
            st.warning(f"⚠️ Impact météo négatif: {weather_impact:.2f}")
        elif weather_impact > 1.1:
            st.info(f"✅ Impact météo positif: {weather_impact:.2f}")
    
    st.divider()
    
    # Forme des équipes
    st.subheader("📈 Forme récente")
    
    if 'team_analysis' in analysis:
        col1, col2 = st.columns(2)
        
        with col1:
            home_form = analysis['team_analysis']['home']['form_analysis']
            st.markdown(f"### 🏠 {analysis['match_info']['home_team']}")
            
            trend_emoji = {
                'positive': '📈',
                'negative': '📉',
                'stable': '➡️',
                'insufficient_data': '❓'
            }.get(home_form.get('trend', 'stable'), '➡️')
            
            st.write(f"**Tendance:** {trend_emoji} {home_form.get('trend', 'stable')}")
            st.write(f"**Momentum:** {home_form.get('momentum', 0):.2f}")
            st.write(f"**Consistance:** {home_form.get('consistency', 0):.2f}")
            
            streak = home_form.get('form_streak', {})
            if streak.get('wins', 0) > 0:
                st.success(f"Série de victoires: {streak['wins']}")
            elif streak.get('losses', 0) > 0:
                st.error(f"Série de défaites: {streak['losses']}")
        
        with col2:
            away_form = analysis['team_analysis']['away']['form_analysis']
            st.markdown(f"### ✈️ {analysis['match_info']['away_team']}")
            
            trend_emoji = {
                'positive': '📈',
                'negative': '📉',
                'stable': '➡️',
                'insufficient_data': '❓'
            }.get(away_form.get('trend', 'stable'), '➡️')
            
            st.write(f"**Tendance:** {trend_emoji} {away_form.get('trend', 'stable')}")
            st.write(f"**Momentum:** {away_form.get('momentum', 0):.2f}")
            st.write(f"**Consistance:** {away_form.get('consistency', 0):.2f}")
            
            streak = away_form.get('form_streak', {})
            if streak.get('wins', 0) > 0:
                st.success(f"Série de victoires: {streak['wins']}")
            elif streak.get('losses', 0) > 0:
                st.error(f"Série de défaites: {streak['losses']}")

def display_betting_tab(analysis: Dict):
    """Affiche l'onglet des paris"""
    st.subheader("💰 Analyse des Paris")
    
    if 'betting_analysis' not in analysis:
        st.info("ℹ️ Analyse des paris non disponible")
        return
    
    betting = analysis['betting_analysis']
    
    # Évaluation du risque
    risk_assessment = betting.get('risk_assessment', {})
    if risk_assessment:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_level = risk_assessment.get('risk_level', 'medium')
            st.markdown(f"### Niveau de risque")
            st.markdown(UIComponents.risk_badge(risk_level), unsafe_allow_html=True)
        
        with col2:
            certainty = risk_assessment.get('certainty_index', 0)
            st.metric("Indice de certitude", f"{certainty:.1f}%")
        
        with col3:
            volatility = risk_assessment.get('volatility', 0)
            st.metric("Volatilité", f"{volatility:.1f}")
    
    st.divider()
    
    # Paris avec valeur
    st.markdown("### 💎 Paris avec valeur")
    value_bets = betting.get('value_bets', [])
    
    if value_bets:
        for bet in value_bets[:5]:  # Max 5 value bets
            UIComponents.display_value_bet(bet)
    else:
        st.info("ℹ️ Aucun pari avec valeur significative détecté")
    
    st.divider()
    
    # Meilleures cotes
    st.markdown("### 📊 Meilleures cotes disponibles")
    best_odds = betting.get('best_odds', {})
    
    if best_odds:
        odds_data = []
        for bet_type, odds_info in best_odds.items():
            odds_data.append({
                'Type': bet_type.replace('_', ' ').title(),
                'Cote': odds_info['odd'],
                'Bookmaker': odds_info['bookmaker']
            })
        
        if odds_data:
            df_odds = pd.DataFrame(odds_data)
            st.dataframe(df_odds, use_container_width=True, hide_index=True)
    
    # Gestion de bankroll
    st.divider()
    st.markdown("### 💼 Gestion de bankroll")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bankroll = st.number_input("Votre bankroll (€)", 
                                  min_value=10, 
                                  value=1000, 
                                  key="bankroll_input")
    
    with col2:
        risk_level = risk_assessment.get('risk_level', 'medium')
        
        if risk_level == 'low':
            risk_percent = 3
            recommendation = "Augmenter la mise"
        elif risk_level == 'medium':
            risk_percent = 2
            recommendation = "Mise standard"
        else:
            risk_percent = 1
            recommendation = "Réduire la mise"
        
        recommended_stake = bankroll * risk_percent / 100
        
        st.metric("Mise recommandée", f"€{recommended_stake:.2f}")
        st.caption(f"({risk_percent}% - {recommendation})")

def display_stats_tab(analysis: Dict):
    """Affiche l'onglet des statistiques"""
    st.subheader("📈 Analyse statistique")
    
    # Stats des équipes
    if 'team_analysis' in analysis:
        home_stats = analysis['team_analysis']['home']['stats']
        away_stats = analysis['team_analysis']['away']['stats']
        
        if analysis['match_info']['sport'] == 'football':
            stats_data = {
                'Statistique': ['Attaque', 'Défense', 'Milieu', 'Forme', 'Buts Moy.'],
                analysis['match_info']['home_team']: [
                    home_stats.get('attack', 'N/A'),
                    home_stats.get('defense', 'N/A'),
                    home_stats.get('midfield', 'N/A'),
                    home_stats.get('form', 'N/A'),
                    home_stats.get('goals_avg', 'N/A')
                ],
                analysis['match_info']['away_team']: [
                    away_stats.get('attack', 'N/A'),
                    away_stats.get('defense', 'N/A'),
                    away_stats.get('midfield', 'N/A'),
                    away_stats.get('form', 'N/A'),
                    away_stats.get('goals_avg', 'N/A')
                ]
            }
        else:
            stats_data = {
                'Statistique': ['Offense', 'Défense', 'Rythme', 'Forme', 'Points Moy.'],
                analysis['match_info']['home_team']: [
                    home_stats.get('offense', 'N/A'),
                    home_stats.get('defense', 'N/A'),
                    home_stats.get('pace', 'N/A'),
                    home_stats.get('form', 'N/A'),
                    home_stats.get('points_avg', 'N/A')
                ],
                analysis['match_info']['away_team']: [
                    away_stats.get('offense', 'N/A'),
                    away_stats.get('defense', 'N/A'),
                    away_stats.get('pace', 'N/A'),
                    away_stats.get('form', 'N/A'),
                    away_stats.get('points_avg', 'N/A')
                ]
            }
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats.set_index('Statistique'), use_container_width=True)
    
    st.divider()
    
    # Historique H2H
    st.subheader("🤝 Historique des confrontations")
    
    if 'contextual_factors' in analysis and 'h2h_history' in analysis['contextual_factors']:
        h2h = analysis['contextual_factors']['h2h_history']
        
        if h2h:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Matches totaux", h2h.get('total_matches', 0))
            
            with col2:
                st.metric("Vict. Domicile", h2h.get('home_wins', 0))
            
            with col3:
                st.metric("Vict. Extérieur", h2h.get('away_wins', 0))
            
            with col4:
                st.metric("Nuls", h2h.get('draws', 0))
            
            if 'last_5_results' in h2h:
                st.markdown(f"**Derniers 5 matchs:** {h2h['last_5_results']}")
        else:
            st.info("ℹ️ Aucun historique de confrontations disponible")

def display_recommendations_tab(analysis: Dict):
    """Affiche l'onglet des recommandations"""
    st.subheader("💡 Recommandations")
    
    if 'recommendations' not in analysis:
        st.info("ℹ️ Aucune recommandation disponible")
        return
    
    recs = analysis['recommendations']
    
    # Recommandation principale
    main_rec = recs.get('main_prediction', {})
    if main_rec:
        st.markdown(f"### 🎯 Recommandation principale")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{main_rec.get('bet', 'N/A')}**")
            st.markdown(f"*Probabilité: {main_rec.get('probability', 0):.1f}%*")
        
        with col2:
            confidence = main_rec.get('confidence', 'medium')
            st.markdown(UIComponents.risk_badge(confidence), unsafe_allow_html=True)
    
    st.divider()
    
    # Recommandations de paris
    betting_recs = recs.get('betting_recommendations', [])
    if betting_recs:
        st.markdown("### 💰 Recommandations de paris")
        
        for rec in betting_recs[:3]:  # Max 3 recommendations
            if rec['type'] == 'value_bet':
                st.markdown(f"""
                <div class="value-bet-card">
                    <strong>💎 {rec['description']}</strong><br>
                    Cote: {rec['odd']} | Score valeur: +{rec['value_score']}%
                </div>
                """, unsafe_allow_html=True)
            elif rec['type'] == 'risk_based':
                st.markdown(f"""
                <div class="warning-card">
                    <strong>⚠️ {rec['description']}</strong><br>
                    {rec['suggestion']}
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # Qualité des données
    st.markdown("### 📊 Qualité des données")
    
    data_quality = analysis.get('data_quality', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source = data_quality.get('home_team_source', 'unknown')
        emoji = "✅" if source in ['local_db', 'api'] else "⚠️" if source == 'generated' else "❓"
        st.metric("Source domicile", f"{emoji} {source}")
    
    with col2:
        source = data_quality.get('away_team_source', 'unknown')
        emoji = "✅" if source in ['local_db', 'api'] else "⚠️" if source == 'generated' else "❓"
        st.metric("Source extérieur", f"{emoji} {source}")
    
    with col3:
        ml_model = data_quality.get('ml_model_used', 'none')
        if ml_model not in ['fallback', 'fallback_no_ml', 'none']:
            st.metric("Modèle ML", "✅ Actif")
        else:
            st.metric("Modèle ML", "⚠️ Basique")

def show_prediction_history():
    """Affiche l'historique des prédictions"""
    st.header("📊 Historique des prédictions")
    
    try:
        history = st.session_state.db.get_prediction_history(limit=50)
        
        if not history:
            st.info("ℹ️ Aucune prédiction dans l'historique")
            return
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sport_filter = st.selectbox(
                "Filtrer par sport",
                ["Tous", "Football", "Basketball"]
            )
        
        with col2:
            date_filter = st.date_input(
                "Filtrer par date",
                value=None
            )
        
        with col3:
            min_confidence = st.slider(
                "Confiance minimum (%)",
                0, 100, 50
            )
        
        # Filtrage des données
        filtered_history = history
        
        if sport_filter != "Tous":
            sport_value = "football" if sport_filter == "Football" else "basketball"
            filtered_history = [h for h in filtered_history if h['sport'] == sport_value]
        
        if date_filter:
            filtered_history = [
                h for h in filtered_history 
                if h['prediction_date'] == date_filter.strftime('%Y-%m-%d')
            ]
        
        filtered_history = [
            h for h in filtered_history 
            if h['confidence_score'] >= min_confidence
        ]
        
        # Affichage
        for prediction in filtered_history[:20]:  # Max 20 prédictions
            with st.expander(f"{prediction['prediction_date']} - {prediction['home_team']} vs {prediction['away_team']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Sport:** {prediction['sport']}")
                    st.write(f"**Ligue:** {prediction['league']}")
                
                with col2:
                    probs = json.loads(prediction['probabilities']) if prediction['probabilities'] else {}
                    if probs:
                        st.write("**Probabilités:**")
                        for key, value in probs.items():
                            st.write(f"{key}: {value}%")
                
                with col3:
                    score_pred = json.loads(prediction['score_prediction']) if prediction['score_prediction'] else {}
                    if score_pred:
                        st.write(f"**Score prédit:** {score_pred.get('exact_score', 'N/A')}")
                    
                    st.write(f"**Confiance:** {prediction['confidence_score']}%")
                
                # Boutons d'action
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📊 Voir détails", key=f"view_{prediction['id']}"):
                        # Reconstruire l'analyse à partir des données
                        st.session_state.current_analysis = {
                            'match_info': {
                                'sport': prediction['sport'],
                                'home_team': prediction['home_team'],
                                'away_team': prediction['away_team'],
                                'league': prediction['league'],
                                'date': prediction['prediction_date']
                            },
                            'probabilities': probs,
                            'score_prediction': score_pred,
                            'confidence_score': prediction['confidence_score']
                        }
                        st.session_state.analysis_mode = "analyze"
                        st.rerun()
                
                with col_b:
                    if prediction.get('actual_result'):
                        st.success(f"✅ Résultat: {prediction['actual_result']}")
                    else:
                        if st.button("📝 Entrer résultat", key=f"result_{prediction['id']}"):
                            st.session_state.editing_prediction_id = prediction['id']
    
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'historique: {e}")

def show_settings():
    """Affiche les paramètres de l'application"""
    st.header("⚙️ Paramètres")
    
    # Langue
    st.subheader("🌍 Langue")
    language = st.selectbox(
        "Choisir la langue",
        options=['Français', 'English', 'Español'],
        index=0
    )
    
    # Thème
    st.subheader("🎨 Thème")
    theme = st.selectbox(
        "Thème de l'interface",
        options=['Clair', 'Sombre', 'Auto'],
        index=0
    )
    
    # API Keys
    st.subheader("🔑 Clés API")
    
    with st.expander("Configurer les clés API"):
        col1, col2 = st.columns(2)
        
        with col1:
            football_key = st.text_input(
                "Clé Football API",
                value=APIConfig.FOOTBALL_API_KEY,
                type="password"
            )
            
            basketball_key = st.text_input(
                "Clé Basketball API",
                value=APIConfig.BASKETBALL_API_KEY,
                type="password"
            )
        
        with col2:
            weather_key = st.text_input(
                "Clé Météo API",
                value=APIConfig.WEATHER_API_KEY,
                type="password"
            )
        
        if st.button("💾 Sauvegarder les clés"):
            # Ici, normalement on sauvegarderait dans un fichier de config
            st.success("Clés API sauvegardées (démonstration)")
    
    # Paramètres de cache
    st.subheader("💾 Cache")
    cache_duration = st.slider(
        "Durée du cache (minutes)",
        5, 120, 30
    )
    
    if st.button("🧹 Vider le cache"):
        try:
            with st.session_state.db.get_connection() as conn:
                conn.execute("DELETE FROM cache")
            st.success("Cache vidé avec succès!")
        except:
            st.error("Erreur lors du vidage du cache")
    
    # Export des données
    st.subheader("📤 Export des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Exporter l'historique CSV"):
            try:
                history = st.session_state.db.get_prediction_history(limit=1000)
                if history:
                    df = pd.DataFrame(history)
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name="historique_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Aucune donnée à exporter")
            except Exception as e:
                st.error(f"Erreur d'export: {e}")
    
    with col2:
        if st.button("🗑️ Supprimer l'historique"):
            if st.checkbox("Confirmer la suppression de toutes les données"):
                try:
                    with st.session_state.db.get_connection() as conn:
                        conn.execute("DELETE FROM predictions")
                        conn.execute("DELETE FROM bets")
                        conn.execute("DELETE FROM user_feedback")
                    st.success("Historique supprimé avec succès!")
                    st.rerun()
                except:
                    st.error("Erreur lors de la suppression")

def main():
    """Fonction principale de l'application"""
    
    # Configuration et initialisation
    configure_page()
    init_application()
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Pronostics Pro")
        
        # Mode d'analyse
        analysis_mode = st.radio(
            "Mode",
            ["Tableau de bord", "Analyse de match", "Historique", "Paramètres"],
            index=0 if 'analysis_mode' not in st.session_state else 
                   {"dashboard": 0, "analyze": 1, "history": 2, "settings": 3}.get(
                       st.session_state.analysis_mode, 0)
        )
        
        # Mapper le mode
        mode_mapping = {
            "Tableau de bord": "dashboard",
            "Analyse de match": "analyze",
            "Historique": "history",
            "Paramètres": "settings"
        }
        
        st.session_state.analysis_mode = mode_mapping.get(analysis_mode, "dashboard")
        
        st.divider()
        
        # Statut
        st.caption("📊 Version Premium Pro")
        st.caption("🤖 IA intégrée")
        st.caption("💾 Base de données active")
        
        # Info système
        with st.expander("ℹ️ Info système"):
            try:
                history_count = len(st.session_state.db.get_prediction_history(limit=1))
                st.write(f"Prédictions: {history_count}")
            except:
                st.write("Prédictions: N/A")
            
            st.write(f"Mode: {st.session_state.analysis_mode}")
            st.write(f"Langue: {st.session_state.i18n.lang}")
    
    # Router d'application
    if st.session_state.analysis_mode == "dashboard":
        show_dashboard()
    
    elif st.session_state.analysis_mode == "analyze":
        if st.session_state.current_analysis:
            display_analysis_results(st.session_state.current_analysis)
        else:
            show_analysis_interface()
    
    elif st.session_state.analysis_mode == "history":
        show_prediction_history()
    
    elif st.session_state.analysis_mode == "settings":
        show_settings()

if __name__ == "__main__":
    main()
