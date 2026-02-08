# app.py - Système de Pronostics Multi-Sports Ultra-Précis
# Version améliorée avec analyse avancée des scores exacts

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import random
import time
import json
from typing import Dict, List, Optional, Tuple, Any
import warnings
from collections import defaultdict
import math
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
                'goal_frequency': 2.8,
                'poisson_lambda_base': 1.4,
                'clean_sheet_prob': 0.25
            },
            'score_ranges': {
                'low_scoring': (0, 2),
                'medium_scoring': (3, 4),
                'high_scoring': (5, float('inf'))
            }
        },
        'basketball': {
            'name': 'Basketball 🏀',
            'icon': '🏀',
            'team_size': 5,
            'duration': 48,
            'scoring_type': 'points',
            'periods': ['Q1', 'Q2', 'Q3', 'Q4'],
            'factors': {
                'home_advantage': 1.10,
                'draw_probability': 0.01,
                'point_frequency': 200,
                'std_dev_multiplier': 0.12,
                'clutch_factor': 1.05
            },
            'score_ranges': {
                'low_scoring': (0, 160),
                'medium_scoring': (161, 210),
                'high_scoring': (211, float('inf'))
            }
        }
    }

# =============================================================================
# COLLECTEUR DE DONNÉES MULTI-SPORTS AMÉLIORÉ
# =============================================================================

class AdvancedDataCollector:
    """Collecteur de données avancé avec historique de scores"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 1800
        
        # Bases de données étendues
        self.football_data = self._init_advanced_football_data()
        self.basketball_data = self._init_advanced_basketball_data()
        
        # Historique de scores
        self.score_history = {
            'football': defaultdict(list),
            'basketball': defaultdict(list)
        }
    
    def _init_advanced_football_data(self):
        """Données football avancées avec historiques de scores"""
        teams_data = {
            'Paris SG': {
                'attack': 96, 'defense': 89, 'midfield': 92, 
                'form': 'WWDLW', 'goals_avg': 2.4,
                'home_goals_avg': 2.7, 'away_goals_avg': 2.1,
                'clean_sheets': 0.35, 'scoring_frequency': 0.78,
                'recent_scores': ['3-1', '2-0', '1-1', '4-2', '3-0']
            },
            'Marseille': {
                'attack': 85, 'defense': 81, 'midfield': 83,
                'form': 'DWWLD', 'goals_avg': 1.8,
                'home_goals_avg': 2.0, 'away_goals_avg': 1.6,
                'clean_sheets': 0.28, 'scoring_frequency': 0.65,
                'recent_scores': ['2-1', '0-0', '3-2', '1-2', '2-0']
            },
            'Real Madrid': {
                'attack': 97, 'defense': 90, 'midfield': 94,
                'form': 'WDWWW', 'goals_avg': 2.6,
                'home_goals_avg': 2.9, 'away_goals_avg': 2.3,
                'clean_sheets': 0.40, 'scoring_frequency': 0.82,
                'recent_scores': ['3-0', '2-1', '4-1', '1-0', '3-2']
            }
        }
        
        return {
            'teams': teams_data,
            'leagues': {
                'Ligue 1': {
                    'goals_avg': 2.7, 'draw_rate': 0.28,
                    'common_scores': ['1-1', '2-1', '1-0', '2-0', '0-0'],
                    'score_distribution': {
                        '0-0': 0.08, '1-0': 0.12, '2-0': 0.10,
                        '1-1': 0.15, '2-1': 0.14, '2-2': 0.07,
                        '3-0': 0.06, '3-1': 0.08, '3-2': 0.04
                    }
                }
            }
        }
    
    def _init_advanced_basketball_data(self):
        """Données basketball avancées"""
        return {
            'teams': {
                'Boston Celtics': {
                    'offense': 118, 'defense': 110, 'pace': 98,
                    'form': 'WWLWW', 'points_avg': 118.5,
                    'home_points_avg': 120.1, 'away_points_avg': 116.9,
                    'quarter_distribution': [30, 29, 30, 31],
                    'recent_scores': ['121-107', '115-110', '108-112', '125-119', '118-102']
                },
                'LA Lakers': {
                    'offense': 114, 'defense': 115, 'pace': 100,
                    'form': 'WLWLD', 'points_avg': 114.7,
                    'home_points_avg': 116.3, 'away_points_avg': 113.1,
                    'quarter_distribution': [28, 29, 29, 29],
                    'recent_scores': ['112-108', '105-120', '119-117', '110-115', '122-118']
                }
            },
            'leagues': {
                'NBA': {
                    'points_avg': 115.0, 'pace': 99.5, 'home_win_rate': 0.58,
                    'common_totals': [210, 215, 220, 225, 230],
                    'score_variance': 12.5
                }
            }
        }
    
    def get_score_probabilities(self, sport: str, home_team: str, away_team: str, 
                               league: str) -> Dict[str, float]:
        """Calcule les probabilités des scores exacts"""
        if sport == 'football':
            return self._calculate_football_score_probs(home_team, away_team, league)
        else:
            return self._calculate_basketball_score_probs(home_team, away_team, league)
    
    def _calculate_football_score_probs(self, home_team: str, away_team: str, 
                                       league: str) -> Dict[str, float]:
        """Probabilités des scores football"""
        try:
            home_data = self.get_team_data('football', home_team)
            away_data = self.get_team_data('football', away_team)
            league_data = self.get_league_data('football', league)
            
            home_attack = home_data.get('attack', 75)
            home_defense = home_data.get('defense', 75)
            away_attack = away_data.get('attack', 75)
            away_defense = away_data.get('defense', 75)
            
            # Lambda Poisson ajusté
            home_lambda = (home_attack / 100) * ((100 - away_defense) / 100) * 2.5 * 1.2
            away_lambda = (away_attack / 100) * ((100 - home_defense) / 100) * 2.0
            
            # Ajustement ligue
            league_factor = league_data.get('goals_avg', 2.7) / 2.7
            home_lambda *= league_factor
            away_lambda *= league_factor
            
            # Calcul des probabilités pour chaque score possible
            max_goals = 5
            score_probs = {}
            
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    # Probabilité Poisson
                    home_prob = self._poisson_pmf(i, home_lambda)
                    away_prob = self._poisson_pmf(j, away_lambda)
                    
                    # Probabilité conjointe
                    joint_prob = home_prob * away_prob
                    
                    if joint_prob > 0.001:  # Seuil minimal
                        score = f"{i}-{j}"
                        score_probs[score] = round(joint_prob * 100, 2)
            
            # Normalisation
            total = sum(score_probs.values())
            if total > 0:
                score_probs = {k: round((v / total) * 100, 2) for k, v in score_probs.items()}
            
            # Tri par probabilité décroissante
            sorted_probs = dict(sorted(score_probs.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:10])  # Top 10
            
            return sorted_probs
            
        except Exception as e:
            # Fallback
            return {
                '1-1': 15.0, '2-1': 12.0, '1-0': 10.0,
                '2-0': 9.0, '0-0': 8.0, '2-2': 7.0,
                '3-1': 6.0, '1-2': 5.0, '3-0': 4.0,
                '0-1': 4.0
            }
    
    def _calculate_basketball_score_probs(self, home_team: str, away_team: str,
                                         league: str) -> Dict[str, float]:
        """Probabilités des scores basketball (intervalles)"""
        try:
            home_data = self.get_team_data('basketball', home_team)
            away_data = self.get_team_data('basketball', away_team)
            league_data = self.get_league_data('basketball', league)
            
            home_offense = home_data.get('offense', 100)
            away_offense = away_data.get('offense', 95)
            home_defense = home_data.get('defense', 100)
            away_defense = away_data.get('defense', 100)
            
            # Points attendus
            home_exp = (home_offense / 100) * ((100 - away_defense) / 100) * 110 * 1.05
            away_exp = (away_offense / 100) * ((100 - home_defense) / 100) * 110
            
            # Écart-type
            std_dev = league_data.get('score_variance', 12.5)
            
            # Intervalles de scores probables
            score_ranges = [
                (f"{int(home_exp-8)}-{int(away_exp-8)}", 15),
                (f"{int(home_exp-4)}-{int(away_exp-4)}", 20),
                (f"{int(home_exp)}-{int(away_exp)}", 25),
                (f"{int(home_exp+4)}-{int(away_exp+4)}", 20),
                (f"{int(home_exp+8)}-{int(away_exp+8)}", 15)
            ]
            
            # Calcul des probabilités
            score_probs = {}
            for score_range, weight in score_ranges:
                score_probs[score_range] = weight
            
            # Normalisation
            total = sum(score_probs.values())
            score_probs = {k: round((v / total) * 100, 2) for k, v in score_probs.items()}
            
            return dict(sorted(score_probs.items(), key=lambda x: x[1], reverse=True))
            
        except:
            return {
                '105-100': 18.0, '108-102': 15.0, '102-98': 12.0,
                '110-105': 15.0, '98-95': 10.0, '115-110': 12.0,
                '100-96': 10.0, '112-108': 8.0
            }
    
    def _poisson_pmf(self, k: int, lam: float) -> float:
        """Fonction de masse de probabilité Poisson"""
        try:
            return (lam ** k) * math.exp(-lam) / math.factorial(k)
        except:
            return 0.0
    
    # Méthodes existantes adaptées...
    def get_team_data(self, sport: str, team_name: str) -> Dict:
        """Adaptée de la version précédente"""
        # Implémentation existante...
        pass
    
    def get_league_data(self, sport: str, league_name: str) -> Dict:
        """Adaptée de la version précédente"""
        # Implémentation existante...
        pass

# =============================================================================
# MOTEUR DE PRÉDICTION AVEC ANALYSE AVANCÉE DES SCORES
# =============================================================================

class AdvancedPredictionEngine:
    """Moteur de prédiction avec analyse approfondie des scores"""
    
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.config = MultiSportConfig()
    
    def predict_match_with_details(self, sport: str, home_team: str, away_team: str,
                                  league: str, match_date: date) -> Dict:
        """Prédiction détaillée avec analyse avancée"""
        base_prediction = self._get_base_prediction(sport, home_team, away_team, league, match_date)
        
        # Analyse avancée des scores
        score_analysis = self._analyze_exact_scores(sport, home_team, away_team, league)
        
        # Combinaison des résultats
        return {
            **base_prediction,
            'advanced_analysis': score_analysis,
            'exact_score_predictions': self._generate_exact_score_predictions(
                sport, home_team, away_team, league, score_analysis
            )
        }
    
    def _analyze_exact_scores(self, sport: str, home_team: str, away_team: str,
                             league: str) -> Dict[str, Any]:
        """Analyse approfondie des scores exacts"""
        if sport == 'football':
            return self._analyze_football_exact_scores(home_team, away_team, league)
        else:
            return self._analyze_basketball_exact_scores(home_team, away_team, league)
    
    def _analyze_football_exact_scores(self, home_team: str, away_team: str,
                                      league: str) -> Dict[str, Any]:
        """Analyse détaillée des scores football"""
        score_probs = self.data_collector.get_score_probabilities(
            'football', home_team, away_team, league
        )
        
        # Analyse des patterns
        analysis = {
            'most_likely_scores': [],
            'clean_sheet_probability': 0,
            'high_scoring_probability': 0,
            'draw_probability': 0,
            'score_trends': [],
            'key_insights': []
        }
        
        # Scores les plus probables
        top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        analysis['most_likely_scores'] = [
            {'score': score, 'probability': prob} 
            for score, prob in top_scores
        ]
        
        # Calcul des probabilités agrégées
        clean_sheet = 0
        high_scoring = 0
        draw = 0
        
        for score_str, prob in score_probs.items():
            home_goals, away_goals = map(int, score_str.split('-'))
            
            # Clean sheet
            if home_goals == 0 or away_goals == 0:
                clean_sheet += prob
            
            # Match à haut score (3+ buts par équipe ou 5+ total)
            if home_goals >= 3 or away_goals >= 3 or (home_goals + away_goals) >= 5:
                high_scoring += prob
            
            # Match nul
            if home_goals == away_goals:
                draw += prob
        
        analysis['clean_sheet_probability'] = round(clean_sheet, 1)
        analysis['high_scoring_probability'] = round(high_scoring, 1)
        analysis['draw_probability'] = round(draw, 1)
        
        # Tendances
        total_goals_dist = self._analyze_total_goals_distribution(score_probs)
        analysis['score_trends'] = total_goals_dist
        
        # Insights
        if top_scores[0][1] > 15:
            analysis['key_insights'].append(f"Score {top_scores[0][0]} très probable ({top_scores[0][1]}%)")
        
        if clean_sheet > 40:
            analysis['key_insights'].append("Forte probabilité qu'une équipe ne marque pas")
        
        if high_scoring > 30:
            analysis['key_insights'].append("Risque de match à haut score")
        
        return analysis
    
    def _analyze_basketball_exact_scores(self, home_team: str, away_team: str,
                                        league: str) -> Dict[str, Any]:
        """Analyse détaillée des scores basketball"""
        score_probs = self.data_collector.get_score_probabilities(
            'basketball', home_team, away_team, league
        )
        
        analysis = {
            'most_likely_ranges': [],
            'high_scoring_probability': 0,
            'close_game_probability': 0,
            'blowout_probability': 0,
            'quarter_analysis': {},
            'key_insights': []
        }
        
        # Plages de scores les plus probables
        top_ranges = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:4]
        analysis['most_likely_ranges'] = [
            {'range': score_range, 'probability': prob}
            for score_range, prob in top_ranges
        ]
        
        # Analyse des scores
        high_scoring = 0
        close_game = 0
        blowout = 0
        
        for score_range, prob in score_probs.items():
            # Extraire les scores moyens de la plage
            try:
                home_pts, away_pts = map(int, score_range.split('-'))
                total = home_pts + away_pts
                diff = abs(home_pts - away_pts)
                
                # Match à haut score
                if total > 220:
                    high_scoring += prob
                
                # Match serré
                if diff <= 5:
                    close_game += prob
                
                # Écart important
                if diff > 15:
                    blowout += prob
            except:
                continue
        
        analysis['high_scoring_probability'] = round(high_scoring, 1)
        analysis['close_game_probability'] = round(close_game, 1)
        analysis['blowout_probability'] = round(blowout, 1)
        
        # Analyse par quart-temps
        analysis['quarter_analysis'] = self._analyze_quarters(home_team, away_team, league)
        
        # Insights
        if close_game > 50:
            analysis['key_insights'].append("Match très serré attendu")
        
        if blowout > 20:
            analysis['key_insights'].append("Risque d'écart important")
        
        return analysis
    
    def _analyze_total_goals_distribution(self, score_probs: Dict[str, float]) -> List[Dict]:
        """Analyse la distribution du total de buts"""
        total_goals_dist = defaultdict(float)
        
        for score_str, prob in score_probs.items():
            home_goals, away_goals = map(int, score_str.split('-'))
            total = home_goals + away_goals
            
            # Regroupement par catégorie
            if total == 0:
                total_goals_dist["0 buts"] += prob
            elif total == 1:
                total_goals_dist["1 but"] += prob
            elif total == 2:
                total_goals_dist["2 buts"] += prob
            elif total == 3:
                total_goals_dist["3 buts"] += prob
            elif total == 4:
                total_goals_dist["4 buts"] += prob
            elif total == 5:
                total_goals_dist["5 buts"] += prob
            else:
                total_goals_dist["6+ buts"] += prob
        
        return [
            {'total': total, 'probability': round(prob, 1)}
            for total, prob in sorted(total_goals_dist.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def _analyze_quarters(self, home_team: str, away_team: str, league: str) -> Dict:
        """Analyse des performances par quart-temps"""
        try:
            home_data = self.data_collector.get_team_data('basketball', home_team)
            away_data = self.data_collector.get_team_data('basketball', away_team)
            
            home_quarters = home_data.get('quarter_distribution', [25, 25, 25, 25])
            away_quarters = away_data.get('quarter_distribution', [25, 25, 25, 25])
            
            return {
                'home_quarters': home_quarters,
                'away_quarters': away_quarters,
                'strong_quarters': self._identify_strong_quarters(home_quarters, away_quarters)
            }
        except:
            return {}
    
    def _identify_strong_quarters(self, home_q: List[int], away_q: List[int]) -> List[str]:
        """Identifie les quarts-temps décisifs"""
        strong_q = []
        
        for i in range(4):
            if home_q[i] > 28 or away_q[i] > 28:
                strong_q.append(f"Q{i+1}")
        
        return strong_q
    
    def _generate_exact_score_predictions(self, sport: str, home_team: str, away_team: str,
                                         league: str, analysis: Dict) -> Dict[str, Any]:
        """Génère des prédictions de scores exacts"""
        if sport == 'football':
            return self._generate_football_exact_predictions(home_team, away_team, league, analysis)
        else:
            return self._generate_basketball_exact_predictions(home_team, away_team, league, analysis)
    
    def _generate_football_exact_predictions(self, home_team: str, away_team: str,
                                            league: str, analysis: Dict) -> Dict[str, Any]:
        """Génère des prédictions football détaillées"""
        score_probs = self.data_collector.get_score_probabilities(
            'football', home_team, away_team, league
        )
        
        return {
            'top_5_scores': analysis['most_likely_scores'],
            'score_probabilities': score_probs,
            'safe_bet': self._find_safe_bet(score_probs),
            'risky_bet': self._find_risky_bet(score_probs),
            'value_bet': self._find_value_bet(score_probs),
            'summary': self._generate_score_summary(score_probs, home_team, away_team)
        }
    
    def _generate_basketball_exact_predictions(self, home_team: str, away_team: str,
                                              league: str, analysis: Dict) -> Dict[str, Any]:
        """Génère des prédictions basketball détaillées"""
        score_probs = self.data_collector.get_score_probabilities(
            'basketball', home_team, away_team, league
        )
        
        return {
            'top_ranges': analysis['most_likely_ranges'],
            'range_probabilities': score_probs,
            'predicted_quarters': self._predict_quarter_scores(home_team, away_team),
            'momentum_shifts': self._analyze_momentum(home_team, away_team),
            'clutch_analysis': self._analyze_clutch_performance(home_team, away_team),
            'summary': self._generate_basketball_summary(score_probs, home_team, away_team)
        }
    
    def _find_safe_bet(self, score_probs: Dict[str, float]) -> Dict:
        """Trouve le pari le plus sûr"""
        safe_scores = []
        
        for score, prob in score_probs.items():
            home_goals, away_goals = map(int, score.split('-'))
            
            # Scores typiques et probables
            if prob > 8 and abs(home_goals - away_goals) <= 2:
                safe_scores.append((score, prob))
        
        if safe_scores:
            safe_scores.sort(key=lambda x: x[1], reverse=True)
            best_score, best_prob = safe_scores[0]
            
            return {
                'score': best_score,
                'probability': best_prob,
                'reason': f"Score équilibré avec haute probabilité ({best_prob}%)"
            }
        
        # Fallback
        most_probable = max(score_probs.items(), key=lambda x: x[1])
        return {
            'score': most_probable[0],
            'probability': most_probable[1],
            'reason': "Score le plus probable"
        }
    
    def _find_risky_bet(self, score_probs: Dict[str, float]) -> Dict:
        """Trouve un pari risqué mais avec valeur"""
        risky_scores = []
        
        for score, prob in score_probs.items():
            home_goals, away_goals = map(int, score.split('-'))
            total = home_goals + away_goals
            
            # Scores rares mais possibles
            if 5 <= prob <= 12 and total >= 4:
                risky_scores.append((score, prob))
        
        if risky_scores:
            risky_scores.sort(key=lambda x: x[0].count('-'), reverse=True)
            best_score, best_prob = risky_scores[0]
            
            return {
                'score': best_score,
                'probability': best_prob,
                'reason': f"Score à haut risque mais avec probabilité significative",
                'odds_estimate': round(100 / best_prob, 1)
            }
        
        return {'score': '3-2', 'probability': 4.0, 'reason': "Pari risqué standard"}
    
    def _find_value_bet(self, score_probs: Dict[str, float]) -> Dict:
        """Trouve un pari avec bonne valeur (probabilité sous-estimée)"""
        value_scores = []
        
        for score, prob in score_probs.items():
            home_goals, away_goals = map(int, score.split('-'))
            
            # Scores qui pourraient surprendre
            if 6 <= prob <= 10 and home_goals != away_goals:
                implied_odds = 100 / prob
                fair_odds = implied_odds * 0.8  # On estime que les cotes sont trop élevées
                
                if fair_odds > 8:  # Bonne valeur
                    value_scores.append((score, prob, fair_odds))
        
        if value_scores:
            value_scores.sort(key=lambda x: x[2], reverse=True)  # Tri par valeur
            best_score, best_prob, best_odds = value_scores[0]
            
            return {
                'score': best_score,
                'probability': best_prob,
                'fair_odds': round(best_odds, 1),
                'reason': f"Probabilité sous-estimée, bonne valeur"
            }
        
        return {'score': '2-1', 'probability': 12.0, 'reason': "Valeur standard"}
    
    def _generate_score_summary(self, score_probs: Dict[str, float], 
                               home_team: str, away_team: str) -> str:
        """Génère un résumé des prédictions de score"""
        top_3 = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary_parts = [
            f"**Analyse des scores exacts {home_team} vs {away_team}**:"
        ]
        
        for i, (score, prob) in enumerate(top_3, 1):
            home_goals, away_goals = map(int, score.split('-'))
            
            if home_goals > away_goals:
                result = f"victoire de {home_team}"
            elif away_goals > home_goals:
                result = f"victoire de {away_team}"
            else:
                result = "match nul"
            
            summary_parts.append(
                f"{i}. **{score}** ({prob}%) - {result} "
                f"({home_goals+away_goals} buts total)"
            )
        
        # Analyse supplémentaire
        draw_prob = sum(prob for score, prob in score_probs.items() 
                       if score.split('-')[0] == score.split('-')[1])
        
        if draw_prob > 25:
            summary_parts.append(f"📊 **Fort risque de match nul** ({draw_prob:.1f}%)")
        
        high_scoring_prob = sum(prob for score, prob in score_probs.items() 
                              if sum(map(int, score.split('-'))) >= 4)
        
        if high_scoring_prob > 35:
            summary_parts.append(f"⚡ **Potentiel de match à haut score** ({high_scoring_prob:.1f}%)")
        
        return "\n\n".join(summary_parts)
    
    def _predict_quarter_scores(self, home_team: str, away_team: str) -> List[Dict]:
        """Prédit les scores par quart-temps"""
        try:
            home_data = self.data_collector.get_team_data('basketball', home_team)
            away_data = self.data_collector.get_team_data('basketball', away_team)
            
            home_q = home_data.get('quarter_distribution', [25, 25, 25, 25])
            away_q = away_data.get('quarter_distribution', [25, 25, 25, 25])
            
            quarters = []
            for i in range(4):
                home_score = home_q[i] + random.randint(-3, 3)
                away_score = away_q[i] + random.randint(-3, 3)
                
                quarters.append({
                    'quarter': f"Q{i+1}",
                    'home': home_score,
                    'away': away_score,
                    'total': home_score + away_score,
                    'momentum': "Équilibre" if abs(home_score - away_score) <= 2 else 
                               f"{home_team}" if home_score > away_score else f"{away_team}"
                })
            
            return quarters
        except:
            return []
    
    def _analyze_momentum(self, home_team: str, away_team: str) -> List[str]:
        """Analyse les changements de momentum possibles"""
        insights = []
        
        # Insights basés sur les données des équipes
        home_data = self.data_collector.get_team_data('basketball', home_team)
        away_data = self.data_collector.get_team_data('basketball', away_team)
        
        home_form = home_data.get('form', '')
        away_form = away_data.get('form', '')
        
        if 'WW' in home_form:
            insights.append(f"**{home_team} en série positive** - Peut prendre un bon départ")
        
        if 'LL' in away_form:
            insights.append(f"**{away_team} en difficulté** - Risque d'écart en première mi-temps")
        
        # Analyse défensive
        if home_data.get('defense', 100) < 105:
            insights.append(f"**{home_team} solide en défense** - Peut contrôler le rythme")
        
        if away_data.get('offense', 95) > 110:
            insights.append(f"**{away_team} offensive puissante** - Peut créer des écarts rapides")
        
        return insights
    
    def _analyze_clutch_performance(self, home_team: str, away_team: str) -> Dict:
        """Analyse la performance en fin de match"""
        home_data = self.data_collector.get_team_data('basketball', home_team)
        away_data = self.data_collector.get_team_data('basketball', away_team)
        
        home_clutch = home_data.get('quarter_distribution', [0, 0, 0, 0])[-1]
        away_clutch = away_data.get('quarter_distribution', [0, 0, 0, 0])[-1]
        
        clutch_diff = home_clutch - away_clutch
        
        if clutch_diff > 3:
            clutch_advantage = home_team
            reason = f"Meilleure performance en 4ème quart-temps ({clutch_diff} points d'avantage)"
        elif clutch_diff < -3:
            clutch_advantage = away_team
            reason = f"Meilleure performance en fin de match ({abs(clutch_diff)} points d'avantage)"
        else:
            clutch_advantage = "Équilibre"
            reason = "Pas d'avantage significatif en fin de match"
        
        return {
            'advantage': clutch_advantage,
            'reason': reason,
            'home_q4': home_clutch,
            'away_q4': away_clutch
        }
    
    def _generate_basketball_summary(self, score_probs: Dict[str, float],
                                    home_team: str, away_team: str) -> str:
        """Génère un résumé pour le basketball"""
        summary_parts = [
            f"**Prédiction détaillée {home_team} vs {away_team}**:"
        ]
        
        # Plages de scores probables
        top_ranges = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for i, (score_range, prob) in enumerate(top_ranges, 1):
            summary_parts.append(f"{i}. **{score_range}** ({prob}%) - Plage de score probable")
        
        # Analyse du total
        total_points_analysis = self._analyze_total_points(score_probs)
        summary_parts.append(f"\n📈 **Analyse du total de points**: {total_points_analysis}")
        
        # Match serré ou écrasant
        close_game_prob = self._calculate_close_game_probability(score_probs)
        
        if close_game_prob > 50:
            summary_parts.append(f"🤝 **Match très serré attendu** ({close_game_prob}% de match à ±5 points)")
        elif close_game_prob < 30:
            summary_parts.append(f"🏆 **Risque d'écart important** ({100-close_game_prob}% de match à +10 points)")
        
        return "\n\n".join(summary_parts)
    
    def _analyze_total_points(self, score_probs: Dict[str, float]) -> str:
        """Analyse le total de points probable"""
        totals = []
        
        for score_range, prob in score_probs.items():
            try:
                home_pts, away_pts = map(int, score_range.split('-'))
                totals.append((home_pts + away_pts, prob))
            except:
                continue
        
        if not totals:
            return "Total autour de 210 points"
        
        # Moyenne pondérée
        weighted_avg = sum(total * prob for total, prob in totals) / sum(prob for _, prob in totals)
        
        if weighted_avg > 220:
            return f"Match à haut score attendu (~{int(weighted_avg)} points)"
        elif weighted_avg < 190:
            return f"Match défensif attendu (~{int(weighted_avg)} points)"
        else:
            return f"Score moyen attendu (~{int(weighted_avg)} points)"
    
    def _calculate_close_game_probability(self, score_probs: Dict[str, float]) -> float:
        """Calcule la probabilité d'un match serré"""
        close_prob = 0
        
        for score_range, prob in score_probs.items():
            try:
                home_pts, away_pts = map(int, score_range.split('-'))
                if abs(home_pts - away_pts) <= 5:
                    close_prob += prob
            except:
                continue
        
        return round(close_prob, 1)
    
    def _get_base_prediction(self, sport: str, home_team: str, away_team: str,
                            league: str, match_date: date) -> Dict:
        """Méthode de base existante (adaptée)"""
        # Implémentation existante...
        pass

# =============================================================================
# INTERFACE STREAMLIT AMÉLIORÉE
# =============================================================================

def main():
    st.set_page_config(
        page_title="Système de Pronostics Avancé",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialisation
    if 'advanced_collector' not in st.session_state:
        st.session_state.advanced_collector = AdvancedDataCollector()
    
    if 'advanced_engine' not in st.session_state:
        st.session_state.advanced_engine = AdvancedPredictionEngine(st.session_state.advanced_collector)
    
    # CSS amélioré
    st.markdown("""
    <style>
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .probability-bar {
        height: 20px;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 10px;
        margin: 5px 0;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .quarter-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête
    st.markdown('<h1 style="text-align: center; color: #1E88E5;">🎯 Système de Pronostics - Analyse Avancée des Scores</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        sport = st.selectbox(
            "🏆 Sport",
            options=['football', 'basketball'],
            format_func=lambda x: 'Football ⚽' if x == 'football' else 'Basketball 🏀'
        )
        
        league = st.selectbox(
            "🏅 Ligue",
            options=['Ligue 1', 'NBA', 'EuroLeague'] if sport == 'basketball' 
            else ['Ligue 1', 'Premier League', 'La Liga']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("🏠 Domicile", 
                                     value="Paris SG" if sport == 'football' else "Boston Celtics")
        with col2:
            away_team = st.text_input("✈️ Extérieur", 
                                     value="Marseille" if sport == 'football' else "LA Lakers")
        
        match_date = st.date_input("📅 Date", value=date.today())
        
        analysis_depth = st.select_slider(
            "📊 Profondeur d'analyse",
            options=['Basique', 'Standard', 'Avancée', 'Expert'],
            value='Avancée'
        )
        
        if st.button("🔍 Analyser les scores exacts", type="primary", use_container_width=True):
            with st.spinner("Analyse statistique en cours..."):
                prediction = st.session_state.advanced_engine.predict_match_with_details(
                    sport, home_team, away_team, league, match_date
                )
                st.session_state.advanced_prediction = prediction
                st.success("Analyse complète générée!")
    
    # Contenu principal
    if 'advanced_prediction' in st.session_state:
        pred = st.session_state.advanced_prediction
        
        # En-tête du match
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.metric("Sport", "⚽ Football" if pred.get('sport') == 'football' else "🏀 Basketball")
        with col2:
            st.markdown(f"<h2 style='text-align: center;'>{pred.get('match', 'Match')}</h2>", 
                       unsafe_allow_html=True)
            st.caption(f"{pred.get('league', '')} • {pred.get('date', '')}")
        with col3:
            st.metric("Confiance", f"{pred.get('confidence', 0)}%")
        
        st.divider()
        
        # Section principale : Analyse des scores exacts
        st.markdown("## 📈 Analyse Détaillée des Scores Exact")
        
        if pred.get('sport') == 'football':
            self._display_football_score_analysis(pred)
        else:
            self._display_basketball_score_analysis(pred)
        
        # Section : Prédictions de pari
        st.markdown("## 💰 Recommandations de Paris")
        self._display_betting_recommendations(pred)
        
        # Section : Visualisations
        st.markdown("## 📊 Visualisations")
        self._display_visualizations(pred)
    
    else:
        # Écran d'accueil
        self._display_homepage()

def _display_football_score_analysis(self, pred: Dict):
    """Affiche l'analyse football"""
    advanced = pred.get('advanced_analysis', {})
    exact = pred.get('exact_score_predictions', {})
    
    # Top 5 scores
    st.markdown("### 🎯 Scores Exact les Plus Probables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.subheader("Top 3 Scores")
        
        top_scores = exact.get('top_5_scores', [])[:3]
        for score_data in top_scores:
            score = score_data.get('score', '0-0')
            prob = score_data.get('probability', 0)
            
            st.metric(f"Score {score}", f"{prob}%")
            
            # Barre de probabilité
            st.markdown(f'<div class="probability-bar" style="width: {prob}%"></div>', 
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.subheader("Analyse des Scores")
        
        # Probabilités agrégées
        metrics_data = [
            ("Probabilité de 0-0", advanced.get('clean_sheet_probability', 0)),
            ("Match à haut score", advanced.get('high_scoring_probability', 0)),
            ("Match nul", advanced.get('draw_probability', 0))
        ]
        
        for label, value in metrics_data:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.text(label)
            with col_b:
                st.text(f"{value}%")
        
        # Distribution des totaux
        st.subheader("📊 Total de Buts")
        trends = advanced.get('score_trends', [])
        for trend in trends:
            st.text(f"{trend.get('total', '')}: {trend.get('probability', 0)}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Insights
    st.markdown("### 🔍 Insights Clés")
    insights = advanced.get('key_insights', [])
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Tableau complet des probabilités
    with st.expander("📋 Toutes les Probabilités de Score"):
        score_probs = exact.get('score_probabilities', {})
        if score_probs:
            df = pd.DataFrame(list(score_probs.items()), columns=['Score', 'Probabilité (%)'])
            st.dataframe(df.sort_values('Probabilité (%)', ascending=False), 
                        use_container_width=True)

def _display_basketball_score_analysis(self, pred: Dict):
    """Affiche l'analyse basketball"""
    advanced = pred.get('advanced_analysis', {})
    exact = pred.get('exact_score_predictions', {})
    
    # Plages de scores
    st.markdown("### 🎯 Plages de Scores Probables")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.subheader("Meilleures Plages")
        
        top_ranges = exact.get('top_ranges', [])
        for range_data in top_ranges[:3]:
            score_range = range_data.get('range', '0-0')
            prob = range_data.get('probability', 0)
            
            st.metric(f"Plage {score_range}", f"{prob}%")
            
            # Barre de probabilité
            st.markdown(f'<div class="probability-bar" style="width: {prob/2}%"></div>', 
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.subheader("Analyse du Match")
        
        metrics = [
            ("Match serré (±5 pts)", advanced.get('close_game_probability', 0)),
            ("Écart important (>15)", advanced.get('blowout_probability', 0)),
            ("Haut score (>220)", advanced.get('high_scoring_probability', 0))
        ]
        
        for label, value in metrics:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.text(label)
            with col_b:
                st.text(f"{value}%")
        
        # Analyse par quart-temps
        quarter_analysis = advanced.get('quarter_analysis', {})
        strong_q = quarter_analysis.get('strong_quarters', [])
        if strong_q:
            st.text(f"Quarts décisifs: {', '.join(strong_q)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyse par quart-temps
    st.markdown("### ⏱️ Analyse par Quart-temps")
    quarters = exact.get('predicted_quarters', [])
    
    if quarters:
        cols = st.columns(4)
        for i, quarter in enumerate(quarters):
            with cols[i]:
                st.markdown(f'<div class="quarter-box">', unsafe_allow_html=True)
                st.subheader(quarter['quarter'])
                st.metric("Domicile", quarter['home'])
                st.metric("Extérieur", quarter['away'])
                st.caption(f"Total: {quarter['total']}")
                st.caption(f"Momentum: {quarter['momentum']}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Momentum et clutch
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Analyse de Momentum")
        momentum_insights = exact.get('momentum_shifts', [])
        for insight in momentum_insights:
            st.info(insight)
    
    with col2:
        st.markdown("### 🏆 Performance en Fin de Match")
        clutch = exact.get('clutch_analysis', {})
        if clutch:
            st.metric("Avantage fin de match", clutch.get('advantage', 'Aucun'))
            st.caption(clutch.get('reason', ''))
    
    # Insights
    st.markdown("### 🔍 Insights Clés")
    insights = advanced.get('key_insights', [])
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def _display_betting_recommendations(self, pred: Dict):
    """Affiche les recommandations de pari"""
    exact = pred.get('exact_score_predictions', {})
    
    col1, col2, col3 = st.columns(3)
    
    # Pari sûr
    with col1:
        safe_bet = exact.get('safe_bet', {})
        st.markdown('<div class="score-card" style="background: linear-gradient(135deg, #4CAF50, #2E7D32);">', 
                   unsafe_allow_html=True)
        st.subheader("✅ Pari Sûr")
        st.metric("Score", safe_bet.get('score', 'N/A'))
        st.metric("Probabilité", f"{safe_bet.get('probability', 0)}%")
        st.caption(safe_bet.get('reason', ''))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Pari risqué
    with col2:
        risky_bet = exact.get('risky_bet', {})
        st.markdown('<div class="score-card" style="background: linear-gradient(135deg, #FF9800, #F57C00);">', 
                   unsafe_allow_html=True)
        st.subheader("🎲 Pari Risqué")
        st.metric("Score", risky_bet.get('score', 'N/A'))
        st.metric("Probabilité", f"{risky_bet.get('probability', 0)}%")
        if 'odds_estimate' in risky_bet:
            st.metric("Cotes estimées", f"{risky_bet['odds_estimate']}")
        st.caption(risky_bet.get('reason', ''))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Pari valeur
    with col3:
        value_bet = exact.get('value_bet', {})
        st.markdown('<div class="score-card" style="background: linear-gradient(135deg, #2196F3, #1976D2);">', 
                   unsafe_allow_html=True)
        st.subheader("💰 Bonne Valeur")
        st.metric("Score", value_bet.get('score', 'N/A'))
        st.metric("Probabilité", f"{value_bet.get('probability', 0)}%")
        if 'fair_odds' in value_bet:
            st.metric("Cotes justes", f"{value_bet['fair_odds']}")
        st.caption(value_bet.get('reason', ''))
        st.markdown('</div>', unsafe_allow_html=True)

def _display_visualizations(self, pred: Dict):
    """Affiche les visualisations"""
    exact = pred.get('exact_score_predictions', {})
    
    if pred.get('sport') == 'football':
        # Graphique des probabilités de score
        score_probs = exact.get('score_probabilities', {})
        if score_probs:
            df = pd.DataFrame(list(score_probs.items()), 
                             columns=['Score', 'Probabilité'])
            df = df.sort_values('Probabilité', ascending=False).head(8)
            
            st.subheader("📊 Probabilités des Scores (Top 8)")
            st.bar_chart(df.set_index('Score'))
    else:
        # Graphique pour le basket
        range_probs = exact.get('range_probabilities', {})
        if range_probs:
            df = pd.DataFrame(list(range_probs.items()), 
                             columns=['Plage', 'Probabilité'])
            df = df.sort_values('Probabilité', ascending=False)
            
            st.subheader("📊 Probabilités des Plages de Score")
            st.bar_chart(df.set_index('Plage'))
        
        # Graphique des quart-temps
        quarters = exact.get('predicted_quarters', [])
        if quarters:
            quarter_data = []
            for q in quarters:
                quarter_data.append({
                    'Quart': q['quarter'],
                    'Domicile': q['home'],
                    'Extérieur': q['away'],
                    'Total': q['total']
                })
            
            df_quarters = pd.DataFrame(quarter_data)
            st.subheader("⏱️ Progression par Quart-temps")
            st.line_chart(df_quarters.set_index('Quart'))

def _display_homepage(self):
    """Affiche la page d'accueil"""
    st.markdown("""
    ## 🎯 Système d'Analyse Avancée des Scores Exact
    
    ### ✨ Nouvelles Fonctionnalités :
    
    **⚽ Football :**
    - 🔍 **Analyse probabiliste des scores exacts** (Poisson distribution)
    - 📊 **Top 10 des scores les plus probables**
    - 🎯 **Pari sûr, risqué et valeur** identifiés
    - 📈 **Distribution du total de buts**
    - 💡 **Insights sur les tendances de score**
    
    **🏀 Basketball :**
    - 🎯 **Plages de scores probables** avec intervalles de confiance
    - ⏱️ **Analyse détaillée par quart-temps**
    - 📈 **Momentum et changements de rythme**
    - 🏆 **Performance en fin de match (clutch time)**
    - 🤝 **Probabilité de match serré vs écrasant**
    
    ### 📊 Méthodologie :
    
    1. **Modélisation statistique** avancée
    2. **Distribution de Poisson** pour les buts
    3. **Analyse historique** des équipes
    4. **Facteurs contextuels** (domicile/visiteur, forme)
    5. **Ajustement ligue** spécifique
    
    ### 🚀 Comment utiliser :
    
    1. Sélectionnez un sport et une ligue
    2. Entrez les noms des équipes
    3. Choisissez la profondeur d'analyse
    4. Cliquez sur "Analyser les scores exacts"
    
    ### 📈 Exemples d'analyse :
    
    """)
    
    # Exemples
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚽ Analyser Paris SG vs Marseille", use_container_width=True):
            st.session_state.sport = 'football'
            st.session_state.home_team = 'Paris SG'
            st.session_state.away_team = 'Marseille'
            st.session_state.league = 'Ligue 1'
            st.rerun()
    
    with col2:
        if st.button("🏀 Analyser Celtics vs Lakers", use_container_width=True):
            st.session_state.sport = 'basketball'
            st.session_state.home_team = 'Boston Celtics'
            st.session_state.away_team = 'LA Lakers'
            st.session_state.league = 'NBA'
            st.rerun()

# Ajouter les méthodes à la classe main
main._display_football_score_analysis = _display_football_score_analysis
main._display_basketball_score_analysis = _display_basketball_score_analysis
main._display_betting_recommendations = _display_betting_recommendations
main._display_visualizations = _display_visualizations
main._display_homepage = _display_homepage

if __name__ == "__main__":
    main()
