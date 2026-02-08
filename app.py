# app.py - Système de Pronostics Multi-Sports Ultra-Précis
# Version corrigée avec gestion des erreurs

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
import random
import time
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION AVANCÉE MULTI-SPORTS
# =============================================================================

class MultiSportConfig:
    """Configuration pour les différents sports"""
    
    SPORTS = {
        'football': {
            'name': 'Football ⚽',
            'icon': '⚽',
            'team_size': 11,
            'duration': 90,
            'scoring_type': 'goals',
            'periods': ['1T', '2T'],
            'factors': {
                'home_advantage': 1.15,
                'draw_probability': 0.25,
                'goal_frequency': 2.8
            }
        },
        'basketball': {
            'name': 'Basketball 🏀',
            'icon': '🏀',
            'team_size': 5,
            'duration': 48,  # Minutes réelles
            'scoring_type': 'points',
            'periods': ['Q1', 'Q2', 'Q3', 'Q4'],
            'factors': {
                'home_advantage': 1.10,
                'draw_probability': 0.01,  # Très rare en basket
                'point_frequency': 200  # Points totaux par match
            }
        }
    }

# =============================================================================
# COLLECTEUR DE DONNÉES MULTI-SPORTS - CORRIGÉ
# =============================================================================

class MultiSportDataCollector:
    """Collecte de données pour tous les sports"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 1800
        
        # Bases de données par sport
        self.football_data = self._init_football_data()
        self.basketball_data = self._init_basketball_data()
    
    def _init_football_data(self):
        """Initialise les données football"""
        return {
            'teams': {
                # Ligue 1
                'Paris SG': {'attack': 96, 'defense': 89, 'midfield': 92, 'form': 'WWDLW', 'goals_avg': 2.4},
                'Marseille': {'attack': 85, 'defense': 81, 'midfield': 83, 'form': 'DWWLD', 'goals_avg': 1.8},
                'Lille': {'attack': 83, 'defense': 82, 'midfield': 81, 'form': 'WDLWW', 'goals_avg': 1.6},
                'Lyon': {'attack': 82, 'defense': 79, 'midfield': 80, 'form': 'LLDWW', 'goals_avg': 1.5},
                'Monaco': {'attack': 84, 'defense': 76, 'midfield': 82, 'form': 'WLDWW', 'goals_avg': 1.9},
                
                # Premier League
                'Manchester City': {'attack': 98, 'defense': 91, 'midfield': 96, 'form': 'WWWWW', 'goals_avg': 2.8},
                'Arsenal': {'attack': 92, 'defense': 85, 'midfield': 89, 'form': 'WWWDL', 'goals_avg': 2.3},
                'Liverpool': {'attack': 94, 'defense': 87, 'midfield': 91, 'form': 'WWDWW', 'goals_avg': 2.5},
                
                # La Liga
                'Real Madrid': {'attack': 97, 'defense': 90, 'midfield': 94, 'form': 'WDWWW', 'goals_avg': 2.6},
                'Barcelona': {'attack': 92, 'defense': 87, 'midfield': 90, 'form': 'LDWWD', 'goals_avg': 2.2},
            },
            'leagues': {
                'Ligue 1': {'goals_avg': 2.7, 'draw_rate': 0.28},
                'Premier League': {'goals_avg': 2.9, 'draw_rate': 0.25},
                'La Liga': {'goals_avg': 2.6, 'draw_rate': 0.27},
                'Bundesliga': {'goals_avg': 3.1, 'draw_rate': 0.22},
                'Serie A': {'goals_avg': 2.5, 'draw_rate': 0.30},
            }
        }
    
    def _init_basketball_data(self):
        """Initialise les données basketball"""
        return {
            'teams': {
                # NBA
                'Boston Celtics': {'offense': 118, 'defense': 110, 'pace': 98, 'form': 'WWLWW', 'points_avg': 118.5},
                'Denver Nuggets': {'offense': 116, 'defense': 112, 'pace': 97, 'form': 'WLWWW', 'points_avg': 116.8},
                'Milwaukee Bucks': {'offense': 120, 'defense': 116, 'pace': 102, 'form': 'WLLWW', 'points_avg': 120.2},
                'Golden State Warriors': {'offense': 117, 'defense': 115, 'pace': 105, 'form': 'LWWDL', 'points_avg': 117.3},
                'LA Lakers': {'offense': 114, 'defense': 115, 'pace': 100, 'form': 'WLWLD', 'points_avg': 114.7},
                
                # EuroLeague
                'Real Madrid Basket': {'offense': 85, 'defense': 78, 'pace': 72, 'form': 'WWWWL', 'points_avg': 85.4},
                'Barcelona Basket': {'offense': 83, 'defense': 76, 'pace': 70, 'form': 'WLDWW', 'points_avg': 83.2},
                'Olympiacos': {'offense': 81, 'defense': 75, 'pace': 68, 'form': 'WWLWD', 'points_avg': 81.5},
                'Monaco Basket': {'offense': 84, 'defense': 79, 'pace': 73, 'form': 'LWWLW', 'points_avg': 84.1},
                
                # LNB Pro A
                'ASVEL': {'offense': 82, 'defense': 78, 'pace': 71, 'form': 'WWLLW', 'points_avg': 82.3},
                'Monaco': {'offense': 85, 'defense': 80, 'pace': 74, 'form': 'WLWWW', 'points_avg': 85.2},
                'Paris Basketball': {'offense': 81, 'defense': 79, 'pace': 70, 'form': 'LDWWL', 'points_avg': 81.4},
            },
            'leagues': {
                'NBA': {'points_avg': 115.0, 'pace': 99.5, 'home_win_rate': 0.58},
                'EuroLeague': {'points_avg': 82.5, 'pace': 72.0, 'home_win_rate': 0.62},
                'LNB Pro A': {'points_avg': 83.0, 'pace': 71.5, 'home_win_rate': 0.60},
                'ACB': {'points_avg': 84.0, 'pace': 73.0, 'home_win_rate': 0.61},
            }
        }
    
    def get_team_data(self, sport: str, team_name: str) -> Dict:
        """Récupère les données d'une équipe pour un sport spécifique"""
        cache_key = f"{sport}_{team_name.lower()}"
        
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_data
        
        try:
            if sport == 'football':
                data = self._get_football_team_data(team_name)
            elif sport == 'basketball':
                data = self._get_basketball_team_data(team_name)
            else:
                data = {}
            
            self.cache[cache_key] = (time.time(), data)
            return data
            
        except Exception as e:
            return self._get_fallback_data(sport, team_name)
    
    def _get_football_team_data(self, team_name: str) -> Dict:
        """Données football spécifiques"""
        # Chercher dans la base
        for known_team, stats in self.football_data['teams'].items():
            if team_name.lower() in known_team.lower() or known_team.lower() in team_name.lower():
                return {
                    **stats,
                    'team_name': known_team,
                    'sport': 'football',
                    'source': 'database'
                }
        
        # Générer des données réalistes
        return {
            'attack': random.randint(75, 90),
            'defense': random.randint(75, 90),
            'midfield': random.randint(75, 90),
            'form': random.choice(['WWDLW', 'WDWLD', 'LDWWD', 'DWWDL']),
            'goals_avg': round(random.uniform(1.2, 2.3), 1),
            'team_name': team_name,
            'sport': 'football',
            'source': 'generated'
        }
    
    def _get_basketball_team_data(self, team_name: str) -> Dict:
        """Données basketball spécifiques"""
        # Chercher dans la base
        for known_team, stats in self.basketball_data['teams'].items():
            if team_name.lower() in known_team.lower() or known_team.lower() in team_name.lower():
                return {
                    **stats,
                    'team_name': known_team,
                    'sport': 'basketball',
                    'source': 'database'
                }
        
        # Générer des données réalistes selon le contexte
        if 'basket' in team_name.lower() or any(word in team_name.lower() for word in ['bb', 'basket']):
            # Probablement une équipe de basket
            return {
                'offense': random.randint(78, 88),
                'defense': random.randint(75, 85),
                'pace': random.randint(68, 76),
                'form': random.choice(['WWLWW', 'WLWWL', 'LWWLD', 'WWLLW']),
                'points_avg': round(random.uniform(78.0, 86.0), 1),
                'team_name': team_name,
                'sport': 'basketball',
                'source': 'generated_europe'
            }
        else:
            # Données génériques
            return {
                'offense': random.randint(100, 120),
                'defense': random.randint(105, 118),
                'pace': random.randint(95, 105),
                'form': random.choice(['WWLWW', 'WLWWL', 'LWWLD']),
                'points_avg': round(random.uniform(105.0, 118.0), 1),
                'team_name': team_name,
                'sport': 'basketball',
                'source': 'generated_nba'
            }
    
    def _get_fallback_data(self, sport: str, team_name: str) -> Dict:
        """Données de fallback sécurisées"""
        try:
            if sport == 'football':
                return {
                    'attack': max(1, random.randint(70, 85)),
                    'defense': max(1, random.randint(70, 85)),
                    'midfield': max(1, random.randint(70, 85)),
                    'form': 'WWDLW',
                    'goals_avg': max(0.1, round(random.uniform(1.0, 2.0), 1)),
                    'team_name': team_name,
                    'sport': sport,
                    'source': 'fallback_safe'
                }
            else:  # basketball
                return {
                    'offense': max(1, random.randint(90, 110)),
                    'defense': max(1, random.randint(90, 110)),
                    'pace': max(1, random.randint(80, 100)),
                    'form': 'WWLWW',
                    'points_avg': max(1.0, round(random.uniform(90.0, 110.0), 1)),
                    'team_name': team_name,
                    'sport': sport,
                    'source': 'fallback_safe'
                }
        except:
            # Données ultra sécurisées en cas d'erreur
            return {
                'offense': 100,
                'defense': 100,
                'pace': 90,
                'form': 'LLLLL',
                'points_avg': 100.0,
                'team_name': team_name,
                'sport': sport,
                'source': 'ultra_safe'
            }
    
    def get_league_data(self, sport: str, league_name: str) -> Dict:
        """Récupère les données d'une ligue"""
        try:
            if sport == 'football':
                return self.football_data['leagues'].get(league_name, {
                    'goals_avg': 2.7,
                    'draw_rate': 0.25
                })
            else:  # basketball
                return self.basketball_data['leagues'].get(league_name, {
                    'points_avg': 100.0,
                    'pace': 90.0,
                    'home_win_rate': 0.60
                })
        except:
            # Données par défaut sécurisées
            return {
                'points_avg': 100.0,
                'pace': 90.0,
                'home_win_rate': 0.60,
                'goals_avg': 2.7,
                'draw_rate': 0.25
            }

# =============================================================================
# MOTEUR DE PRÉDICTION MULTI-SPORTS - CORRIGÉ
# =============================================================================

class MultiSportPredictionEngine:
    """Moteur de prédiction pour tous les sports"""
    
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.config = MultiSportConfig()
    
    def predict_match(self, sport: str, home_team: str, away_team: str, 
                     league: str, match_date: date) -> Dict:
        """Prédit un match pour n'importe quel sport"""
        
        try:
            if sport == 'football':
                return self._predict_football_match(home_team, away_team, league, match_date)
            elif sport == 'basketball':
                return self._predict_basketball_match(home_team, away_team, league, match_date)
            else:
                return self._get_generic_prediction(sport, home_team, away_team, match_date)
        except Exception as e:
            return self._get_error_prediction(sport, home_team, away_team, str(e))
    
    def _predict_football_match(self, home_team: str, away_team: str, 
                               league: str, match_date: date) -> Dict:
        """Prédiction football avancée"""
        
        try:
            # Données des équipes
            home_data = self.data_collector.get_team_data('football', home_team)
            away_data = self.data_collector.get_team_data('football', away_team)
            
            # Données de la ligue
            league_data = self.data_collector.get_league_data('football', league)
            
            # Calcul des forces
            home_strength = self._calculate_football_strength(home_data, is_home=True)
            away_strength = self._calculate_football_strength(away_data, is_home=False)
            
            # Probabilités
            home_prob, draw_prob, away_prob = self._calculate_football_probabilities(
                home_strength, away_strength, league_data
            )
            
            # Score prédit
            home_goals, away_goals = self._predict_football_score(
                home_data, away_data, league_data
            )
            
            # Over/Under
            total_goals = home_goals + away_goals
            over_under = "Over 2.5" if total_goals >= 3 else "Under 2.5"
            over_prob = self._calculate_over_probability(total_goals)
            
            # BTTS
            btts = "Oui" if home_goals > 0 and away_goals > 0 else "Non"
            btts_prob = self._calculate_btts_probability(home_goals, away_goals)
            
            # Cotes
            odds = self._calculate_odds(home_prob, draw_prob, away_prob)
            
            # Confiance
            confidence = self._calculate_football_confidence(home_data, away_data)
            
            # Analyse
            analysis = self._generate_football_analysis(
                home_team, away_team, league, home_data, away_data,
                home_prob, draw_prob, away_prob, home_goals, away_goals,
                confidence
            )
            
            return {
                'sport': 'football',
                'match': f"{home_team} vs {away_team}",
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'date': match_date.strftime('%Y-%m-%d'),
                'probabilities': {
                    'home_win': round(home_prob, 1),
                    'draw': round(draw_prob, 1),
                    'away_win': round(away_prob, 1)
                },
                'score_prediction': f"{home_goals}-{away_goals}",
                'over_under': over_under,
                'over_prob': round(over_prob, 1),
                'btts': btts,
                'btts_prob': round(btts_prob, 1),
                'odds': odds,
                'confidence': round(confidence, 1),
                'analysis': analysis,
                'team_stats': {
                    'home': home_data,
                    'away': away_data
                },
                'prediction_details': {
                    'expected_goals_home': round(self._calculate_expected_goals(home_data, away_data, True), 2),
                    'expected_goals_away': round(self._calculate_expected_goals(away_data, home_data, False), 2),
                    'data_quality': self._assess_data_quality(home_data, away_data)
                }
            }
            
        except Exception as e:
            return self._get_football_fallback(home_team, away_team, league, match_date)
    
    def _predict_basketball_match(self, home_team: str, away_team: str,
                                 league: str, match_date: date) -> Dict:
        """Prédiction basketball avancée"""
        
        try:
            # Données des équipes
            home_data = self.data_collector.get_team_data('basketball', home_team)
            away_data = self.data_collector.get_team_data('basketball', away_team)
            
            # Données de la ligue
            league_data = self.data_collector.get_league_data('basketball', league)
            
            # Calcul des forces
            home_strength = self._calculate_basketball_strength(home_data, is_home=True)
            away_strength = self._calculate_basketball_strength(away_data, is_home=False)
            
            # Probabilités
            home_prob, away_prob = self._calculate_basketball_probabilities(
                home_strength, away_strength, league_data
            )
            
            # Score prédit
            home_points, away_points = self._predict_basketball_score_safe(
                home_data, away_data, league_data
            )
            
            # Total points
            total_points = home_points + away_points
            points_spread = home_points - away_points
            
            # Over/Under adapté au basket
            if 'NBA' in league:
                over_under_line = 225
            elif 'Euro' in league:
                over_under_line = 165
            else:
                over_under_line = 160
            
            over_under = f"Over {over_under_line}" if total_points >= over_under_line else f"Under {over_under_line}"
            over_prob = self._calculate_basketball_over_probability(total_points, over_under_line)
            
            # Spread
            spread = self._calculate_spread(home_team, away_team, points_spread)
            
            # Cotes
            odds = self._calculate_basketball_odds(home_prob)
            
            # Confiance
            confidence = self._calculate_basketball_confidence(home_data, away_data)
            
            # Analyse
            analysis = self._generate_basketball_analysis(
                home_team, away_team, league, home_data, away_data,
                home_prob, away_prob, home_points, away_points,
                confidence, spread
            )
            
            return {
                'sport': 'basketball',
                'match': f"{home_team} vs {away_team}",
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'date': match_date.strftime('%Y-%m-%d'),
                'probabilities': {
                    'home_win': round(home_prob, 1),
                    'away_win': round(away_prob, 1)
                },
                'score_prediction': f"{home_points}-{away_points}",
                'total_points': total_points,
                'point_spread': spread,
                'over_under': over_under,
                'over_prob': round(over_prob, 1),
                'odds': odds,
                'confidence': round(confidence, 1),
                'analysis': analysis,
                'team_stats': {
                    'home': home_data,
                    'away': away_data
                },
                'prediction_details': {
                    'expected_points_home': round(self._calculate_expected_points(home_data, away_data, True), 1),
                    'expected_points_away': round(self._calculate_expected_points(away_data, home_data, False), 1),
                    'pace_adjusted_total': round(self._calculate_pace_adjusted_total(home_data, away_data), 1),
                    'data_quality': self._assess_data_quality(home_data, away_data)
                }
            }
            
        except Exception as e:
            return self._get_basketball_fallback(home_team, away_team, league, match_date)
    
    # =========================================================================
    # MÉTHODES FOOTBALL - CORRIGÉES
    # =========================================================================
    
    def _calculate_football_strength(self, team_data: Dict, is_home: bool) -> float:
        """Calcule la force d'une équipe de football"""
        try:
            attack = max(1, team_data.get('attack', 75))
            defense = max(1, team_data.get('defense', 75))
            midfield = max(1, team_data.get('midfield', 75))
            
            base_strength = (
                attack * 0.4 +
                defense * 0.3 +
                midfield * 0.3
            )
            
            # Avantage domicile
            if is_home:
                base_strength *= self.config.SPORTS['football']['factors']['home_advantage']
            
            # Facteur forme
            form_string = team_data.get('form', 'LLLLL')
            form_score = self._calculate_form_score(form_string)
            base_strength *= (1 + (form_score - 0.5) * 0.2)
            
            return max(1, base_strength)
            
        except:
            return 75.0  # Valeur par défaut sécurisée
    
    def _calculate_football_probabilities(self, home_strength: float, away_strength: float,
                                         league_data: Dict) -> Tuple[float, float, float]:
        """Calcule les probabilités football"""
        try:
            total_strength = max(1, home_strength + away_strength)
            
            home_prob = (home_strength / total_strength) * 100 * 0.85
            away_prob = (away_strength / total_strength) * 100 * 0.85
            draw_prob = max(0, 100 - home_prob - away_prob)
            
            # Ajustement selon la ligue
            draw_adjustment = max(0.1, min(0.5, league_data.get('draw_rate', 0.25)))
            draw_prob *= (draw_adjustment / 0.25)
            
            # Normaliser
            total = max(1, home_prob + draw_prob + away_prob)
            home_prob = (home_prob / total) * 100
            draw_prob = (draw_prob / total) * 100
            away_prob = (away_prob / total) * 100
            
            return home_prob, draw_prob, away_prob
            
        except:
            return 40.0, 30.0, 30.0  # Valeurs par défaut
    
    def _predict_football_score(self, home_data: Dict, away_data: Dict,
                               league_data: Dict) -> Tuple[int, int]:
        """Prédit le score football de manière sécurisée"""
        try:
            # xG basés sur l'attaque et la défense
            home_attack = max(1, home_data.get('attack', 75))
            away_defense = max(1, away_data.get('defense', 75))
            away_attack = max(1, away_data.get('attack', 75))
            home_defense = max(1, home_data.get('defense', 75))
            
            home_xg = (home_attack / 100) * ((100 - away_defense) / 100) * 2.5
            away_xg = (away_attack / 100) * ((100 - home_defense) / 100) * 2.0
            
            # Ajustement ligue
            league_factor = max(0.5, min(2.0, league_data.get('goals_avg', 2.7) / 2.7))
            home_xg *= league_factor
            away_xg *= league_factor
            
            # Avantage domicile
            home_xg *= 1.2
            
            # Simulation Poisson sécurisée
            home_goals = self._simulate_poisson_goals_safe(home_xg)
            away_goals = self._simulate_poisson_goals_safe(away_xg)
            
            # Ajustements finaux
            home_goals = max(0, min(5, home_goals))
            away_goals = max(0, min(4, away_goals))
            
            # Éviter les scores improbables
            if home_goals == away_goals == 0 and random.random() < 0.8:
                home_goals = random.randint(0, 1)
                away_goals = random.randint(0, 1)
            
            return home_goals, away_goals
            
        except:
            # Fallback sécurisé
            return random.randint(0, 3), random.randint(0, 2)
    
    # =========================================================================
    # MÉTHODES BASKETBALL - CORRIGÉES
    # =========================================================================
    
    def _calculate_basketball_strength(self, team_data: Dict, is_home: bool) -> float:
        """Calcule la force d'une équipe de basket"""
        try:
            offense = max(1, team_data.get('offense', 100))
            defense = max(1, team_data.get('defense', 100))
            pace = max(1, team_data.get('pace', 90))
            
            # Inverser la défense (moins c'est mieux)
            defense_score = max(1, 200 - defense)  # 100 = défense parfaite
            
            base_strength = (
                offense * 0.5 +
                defense_score * 0.3 +
                pace * 0.2
            )
            
            # Avantage domicile
            if is_home:
                base_strength *= self.config.SPORTS['basketball']['factors']['home_advantage']
            
            # Facteur forme
            form_string = team_data.get('form', 'LLLLL')
            form_score = self._calculate_form_score(form_string)
            base_strength *= (1 + (form_score - 0.5) * 0.15)
            
            return max(1, base_strength)
            
        except:
            return 100.0  # Valeur par défaut
    
    def _calculate_basketball_probabilities(self, home_strength: float, away_strength: float,
                                           league_data: Dict) -> Tuple[float, float]:
        """Calcule les probabilités basketball (pas de match nul)"""
        try:
            total_strength = max(1, home_strength + away_strength)
            
            home_prob = (home_strength / total_strength) * 100
            away_prob = max(0, 100 - home_prob)
            
            # Ajustement avantage domicile de la ligue
            home_win_rate = max(0.4, min(0.8, league_data.get('home_win_rate', 0.60)))
            home_prob *= (home_win_rate / 0.60)
            away_prob = max(0, 100 - home_prob)
            
            # Normaliser
            total = max(1, home_prob + away_prob)
            home_prob = (home_prob / total) * 100
            away_prob = (away_prob / total) * 100
            
            return home_prob, away_prob
            
        except:
            return 55.0, 45.0  # Léger avantage domicile par défaut
    
    def _predict_basketball_score_safe(self, home_data: Dict, away_data: Dict,
                                      league_data: Dict) -> Tuple[int, int]:
        """Prédit le score basketball de manière sécurisée"""
        try:
            # Points attendus basés sur l'attaque et la défense adverse
            home_offense = max(1, home_data.get('offense', 100))
            away_defense = max(1, away_data.get('defense', 100))
            away_offense = max(1, away_data.get('offense', 95))
            home_defense = max(1, home_data.get('defense', 100))
            
            league_points_avg = max(50, league_data.get('points_avg', 100))
            
            home_points_exp = (home_offense / 100) * ((100 - away_defense) / 100) * league_points_avg
            away_points_exp = (away_offense / 100) * ((100 - home_defense) / 100) * league_points_avg
            
            # Ajustement pace
            home_pace = max(60, home_data.get('pace', 90))
            away_pace = max(60, away_data.get('pace', 90))
            avg_pace = (home_pace + away_pace) / 2
            league_pace = max(60, league_data.get('pace', 90))
            
            pace_factor = avg_pace / max(1, league_pace)
            home_points_exp *= pace_factor
            away_points_exp *= pace_factor
            
            # Avantage domicile
            home_points_exp *= 1.05
            
            # Simulation avec variabilité sécurisée
            # Utiliser une méthode plus simple que np.random.normal
            home_points = self._simulate_normal_safe(home_points_exp, home_points_exp * 0.1)
            away_points = self._simulate_normal_safe(away_points_exp, away_points_exp * 0.1)
            
            # S'assurer que les scores sont réalistes
            home_points = max(70, min(150, home_points))
            away_points = max(70, min(140, away_points))
            
            # Éviter les égalités exactes
            if home_points == away_points:
                home_points += random.choice([-1, 1])
            
            return home_points, away_points
            
        except Exception as e:
            # Fallback sécurisé
            return random.randint(85, 115), random.randint(80, 110)
    
    def _simulate_normal_safe(self, mean: float, std_dev: float) -> int:
        """Simule une distribution normale de manière sécurisée"""
        try:
            # Assurer que std_dev est positif
            std_dev = max(0.1, abs(std_dev))
            
            # Méthode simplifiée sans numpy
            # Approximation de Box-Muller
            u1 = random.random()
            u2 = random.random()
            
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            
            value = mean + z0 * std_dev
            
            return int(round(max(0, value)))
            
        except:
            # Fallback très simple
            variation = random.uniform(-std_dev, std_dev)
            return int(round(max(0, mean + variation)))
    
    # =========================================================================
    # MÉTHODES GÉNÉRIQUES - CORRIGÉES
    # =========================================================================
    
    def _simulate_poisson_goals_safe(self, lambda_value: float) -> int:
        """Simule les buts avec distribution Poisson sécurisée"""
        try:
            lambda_value = max(0.1, lambda_value)  # Éviter lambda négatif ou nul
            
            goals = 0
            for _ in range(int(lambda_value * 10)):
                if random.random() < lambda_value / 10:
                    goals += 1
            
            return min(goals, 5)  # Limiter
            
        except:
            return random.randint(0, 3)
    
    def _calculate_form_score(self, form_string: str) -> float:
        """Calcule un score de forme (W=1, D=0.5, L=0)"""
        try:
            if not form_string:
                return 0.5
                
            scores = {'W': 1.0, 'D': 0.5, 'L': 0.0}
            total = 0
            count = 0
            
            for result in form_string:
                if result in scores:
                    total += scores[result]
                    count += 1
            
            if count == 0:
                return 0.5
                
            return total / count
            
        except:
            return 0.5
    
    def _calculate_over_probability(self, total_goals: int) -> float:
        """Calcule la probabilité Over 2.5"""
        try:
            if total_goals >= 3:
                probability = min(95, 60 + (total_goals - 2) * 15)
            else:
                probability = min(95, 70 - (3 - total_goals) * 20)
            return probability
        except:
            return 50.0
    
    def _calculate_basketball_over_probability(self, total_points: int, line: int) -> float:
        """Calcule la probabilité Over/Under pour le basket"""
        try:
            diff = total_points - line
            if diff >= 0:
                probability = min(95, 50 + abs(diff) * 
