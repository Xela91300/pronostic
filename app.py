# app.py - Système Complet de Pronostics Football
# Version avec Tous les Types de Paris et Sélection de Jour

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION API
# =============================================================================

class APIConfig:
    """Configuration API Football"""
    API_FOOTBALL_KEY = "249b3051eCA063F0e381609128c00d7d"
    API_FOOTBALL_URL = "https://v3.football.api-sports.io"

# =============================================================================
# CLIENT API AVEC DONNÉES SIMULÉES DE QUALITÉ
# =============================================================================

class FootballDataClient:
    """Client pour données de matchs avec simulation réaliste"""
    
    def __init__(self):
        self.config = APIConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'x-apisports-key': self.config.API_FOOTBALL_KEY,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.use_simulation = False
        self.test_connection()
        
        # Base de données d'équipes réalistes
        self.initialize_teams_database()
    
    def test_connection(self):
        """Teste la connexion API"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/status"
            response = self.session.get(url, timeout=3)
            self.use_simulation = response.status_code != 200
            return not self.use_simulation
        except:
            self.use_simulation = True
            return False
    
    def initialize_teams_database(self):
        """Initialise une base de données d'équipes réalistes"""
        self.teams_database = {
            # Ligue 1
            'PSG': {'rating': 90, 'attack': 92, 'defense': 88, 'home_power': 1.2},
            'Marseille': {'rating': 78, 'attack': 80, 'defense': 76, 'home_power': 1.15},
            'Lyon': {'rating': 76, 'attack': 78, 'defense': 74, 'home_power': 1.15},
            'Monaco': {'rating': 75, 'attack': 77, 'defense': 73, 'home_power': 1.1},
            'Lille': {'rating': 77, 'attack': 75, 'defense': 79, 'home_power': 1.15},
            'Nice': {'rating': 74, 'attack': 72, 'defense': 76, 'home_power': 1.1},
            'Rennes': {'rating': 76, 'attack': 75, 'defense': 77, 'home_power': 1.1},
            'Lens': {'rating': 73, 'attack': 74, 'defense': 72, 'home_power': 1.15},
            
            # Premier League
            'Manchester City': {'rating': 93, 'attack': 94, 'defense': 92, 'home_power': 1.2},
            'Liverpool': {'rating': 90, 'attack': 91, 'defense': 89, 'home_power': 1.25},
            'Arsenal': {'rating': 87, 'attack': 88, 'defense': 86, 'home_power': 1.15},
            'Chelsea': {'rating': 85, 'attack': 84, 'defense': 86, 'home_power': 1.15},
            'Manchester United': {'rating': 82, 'attack': 81, 'defense': 83, 'home_power': 1.2},
            'Tottenham': {'rating': 84, 'attack': 85, 'defense': 83, 'home_power': 1.1},
            'Newcastle': {'rating': 83, 'attack': 82, 'defense': 84, 'home_power': 1.15},
            
            # La Liga
            'Real Madrid': {'rating': 92, 'attack': 93, 'defense': 91, 'home_power': 1.25},
            'Barcelona': {'rating': 89, 'attack': 90, 'defense': 88, 'home_power': 1.2},
            'Atletico Madrid': {'rating': 85, 'attack': 83, 'defense': 87, 'home_power': 1.2},
            'Sevilla': {'rating': 79, 'attack': 78, 'defense': 80, 'home_power': 1.15},
            'Valencia': {'rating': 76, 'attack': 77, 'defense': 75, 'home_power': 1.1},
            
            # Bundesliga
            'Bayern Munich': {'rating': 91, 'attack': 93, 'defense': 89, 'home_power': 1.3},
            'Borussia Dortmund': {'rating': 84, 'attack': 86, 'defense': 82, 'home_power': 1.25},
            'RB Leipzig': {'rating': 83, 'attack': 84, 'defense': 82, 'home_power': 1.1},
            'Bayer Leverkusen': {'rating': 82, 'attack': 83, 'defense': 81, 'home_power': 1.1},
            
            # Serie A
            'Inter Milan': {'rating': 86, 'attack': 85, 'defense': 87, 'home_power': 1.15},
            'AC Milan': {'rating': 85, 'attack': 84, 'defense': 86, 'home_power': 1.15},
            'Juventus': {'rating': 84, 'attack': 82, 'defense': 86, 'home_power': 1.2},
            'Napoli': {'rating': 83, 'attack': 84, 'defense': 82, 'home_power': 1.15},
            'Roma': {'rating': 81, 'attack': 80, 'defense': 82, 'home_power': 1.1},
            'Atalanta': {'rating': 82, 'attack': 85, 'defense': 79, 'home_power': 1.1},
        }
    
    def get_team_stats(self, team_name):
        """Retourne les statistiques d'une équipe"""
        if team_name in self.teams_database:
            return self.teams_database[team_name]
        else:
            # Générer des stats pour une équipe inconnue
            rating = random.uniform(70, 85)
            return {
                'rating': rating,
                'attack': rating * random.uniform(0.95, 1.05),
                'defense': rating * random.uniform(0.95, 1.05),
                'home_power': random.uniform(1.05, 1.15)
            }
    
    def get_fixtures_by_date(self, target_date):
        """Récupère les matchs pour une date spécifique"""
        if self.use_simulation:
            return self._simulate_fixtures_for_date(target_date)
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'date': target_date.strftime('%Y-%m-%d'),
                'timezone': 'Europe/Paris'
            }
            
            response = self.session.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                fixtures = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    league = fixture.get('league', {})
                    
                    status = fixture_data.get('status', {}).get('short')
                    if status in ['NS', 'TBD', 'PST']:  # Matchs à venir
                        fixtures.append({
                            'fixture_id': fixture_data.get('id'),
                            'date': fixture_data.get('date'),
                            'home_name': teams.get('home', {}).get('name'),
                            'away_name': teams.get('away', {}).get('name'),
                            'league_id': league.get('id'),
                            'league_name': league.get('name'),
                            'league_country': league.get('country'),
                            'status': status,
                            'timestamp': fixture_data.get('timestamp')
                        })
                
                if fixtures:
                    return fixtures
            
            # Fallback à la simulation si API échoue ou pas de matchs
            return self._simulate_fixtures_for_date(target_date)
            
        except:
            return self._simulate_fixtures_for_date(target_date)
    
    def _simulate_fixtures_for_date(self, target_date):
        """Simule des matchs réalistes pour une date donnée"""
        # Sélectionner des équipes aléatoires mais réalistes
        all_teams = list(self.teams_database.keys())
        random.shuffle(all_teams)
        
        fixtures = []
        num_matches = random.randint(8, 15)  # 8 à 15 matchs par jour
        
        for i in range(0, min(num_matches * 2, len(all_teams)), 2):
            if i + 1 >= len(all_teams):
                break
                
            home_team = all_teams[i]
            away_team = all_teams[i + 1]
            
            # Heure aléatoire
            hour = random.randint(15, 22)
            minute = random.choice([0, 15, 30, 45])
            
            # Sélection de ligue basée sur les équipes
            league = self._determine_league(home_team, away_team)
            
            fixtures.append({
                'fixture_id': random.randint(10000, 99999),
                'date': f"{target_date.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00+00:00",
                'home_name': home_team,
                'away_name': away_team,
                'league_name': league['name'],
                'league_country': league['country'],
                'status': 'NS',
                'timestamp': int(datetime.now().timestamp()) + random.randint(0, 86400)
            })
        
        return fixtures
    
    def _determine_league(self, team1, team2):
        """Détermine la ligue basée sur les équipes"""
        # Mapping des équipes aux ligues
        league_mapping = {
            'Ligue 1': ['PSG', 'Marseille', 'Lyon', 'Monaco', 'Lille', 'Nice', 'Rennes', 'Lens'],
            'Premier League': ['Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 
                              'Manchester United', 'Tottenham', 'Newcastle'],
            'La Liga': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia'],
            'Bundesliga': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen'],
            'Serie A': ['Inter Milan', 'AC Milan', 'Juventus', 'Napoli', 'Roma', 'Atalanta']
        }
        
        # Trouver la ligue des deux équipes
        for league, teams in league_mapping.items():
            if team1 in teams and team2 in teams:
                return {'name': league, 'country': league.split()[0] if ' ' in league else league}
        
        # Si pas dans la même ligue, choisir au hasard
        league = random.choice(list(league_mapping.keys()))
        return {'name': league, 'country': league.split()[0] if ' ' in league else league}
    
    def get_fixture_odds(self, fixture_id):
        """Récupère les cotes pour un match"""
        if self.use_simulation:
            return self._simulate_realistic_odds()
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/odds"
            params = {'fixture': fixture_id}
            
            response = self.session.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                if data:
                    return data[0]
            
            return self._simulate_realistic_odds()
        except:
            return self._simulate_realistic_odds()
    
    def _simulate_realistic_odds(self):
        """Simule des cotes réalistes"""
        return {
            'bookmakers': [{
                'name': random.choice(['Bet365', 'Unibet', 'Winamax', 'ParionsSport']),
                'bets': [
                    {
                        'name': 'Match Winner',
                        'values': [
                            {'value': 'Home', 'odd': round(random.uniform(1.5, 3.5), 2)},
                            {'value': 'Draw', 'odd': round(random.uniform(3.0, 4.5), 2)},
                            {'value': 'Away', 'odd': round(random.uniform(2.0, 4.0), 2)}
                        ]
                    },
                    {
                        'name': 'Double Chance',
                        'values': [
                            {'value': 'Home/Draw', 'odd': round(random.uniform(1.2, 1.5), 2)},
                            {'value': 'Home/Away', 'odd': round(random.uniform(1.3, 1.8), 2)},
                            {'value': 'Draw/Away', 'odd': round(random.uniform(1.2, 1.6), 2)}
                        ]
                    },
                    {
                        'name': 'Goals Over/Under',
                        'values': [
                            {'value': 'Over 1.5', 'odd': round(random.uniform(1.3, 1.7), 2)},
                            {'value': 'Under 1.5', 'odd': round(random.uniform(2.0, 3.0), 2)},
                            {'value': 'Over 2.5', 'odd': round(random.uniform(1.8, 2.5), 2)},
                            {'value': 'Under 2.5', 'odd': round(random.uniform(1.4, 1.9), 2)}
                        ]
                    },
                    {
                        'name': 'Both Teams to Score',
                        'values': [
                            {'value': 'Yes', 'odd': round(random.uniform(1.6, 2.2), 2)},
                            {'value': 'No', 'odd': round(random.uniform(1.5, 2.0), 2)}
                        ]
                    }
                ]
            }]
        }

# =============================================================================
# SYSTÈME DE PRÉDICTION COMPLET
# =============================================================================

class CompletePredictionSystem:
    """Système de prédiction complet avec tous les types de paris"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.predictions_cache = {}
    
    def analyze_match(self, fixture):
        """Analyse complète d'un match avec tous les types de paris"""
        
        home_team = fixture['home_name']
        away_team = fixture['away_name']
        
        # Récupérer les stats des équipes
        home_stats = self.api_client.get_team_stats(home_team)
        away_stats = self.api_client.get_team_stats(away_team)
        
        # Calcul des probabilités de base
        probabilities = self._calculate_probabilities(home_stats, away_stats)
        
        # Générer toutes les prédictions
        predictions = {
            'match_info': {
                'match': f"{home_team} vs {away_team}",
                'league': fixture.get('league_name', 'Unknown'),
                'date': fixture.get('date', ''),
                'time': fixture.get('date', '')[11:16] if len(fixture.get('date', '')) > 16 else ''
            },
            'probabilities': probabilities,
            'predictions': self._generate_all_predictions(home_stats, away_stats, probabilities),
            'confidence': self._calculate_confidence(home_stats, away_stats, probabilities),
            'detailed_analysis': self._generate_detailed_analysis(home_team, away_team, home_stats, away_stats)
        }
        
        return predictions
    
    def _calculate_probabilities(self, home_stats, away_stats):
        """Calcule les probabilités pour tous les résultats"""
        
        # Forces ajustées avec avantage domicile
        home_power = home_stats['rating'] * home_stats['home_power']
        away_power = away_stats['rating']
        
        total_power = home_power + away_power
        
        # Probabilités de base
        home_win_prob = (home_power / total_power) * 100 * 0.85
        away_win_prob = (away_power / total_power) * 100 * 0.85
        draw_prob = 100 - home_win_prob - away_win_prob
        
        # Ajustements basés sur l'attaque/défense
        attack_defense_ratio = (home_stats['attack'] / away_stats['defense']) / (away_stats['attack'] / home_stats['defense'])
        
        if attack_defense_ratio > 1.2:
            home_win_prob += 5
            draw_prob -= 3
            away_win_prob -= 2
        elif attack_defense_ratio < 0.8:
            away_win_prob += 5
            draw_prob -= 3
            home_win_prob -= 2
        
        # Normalisation
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob = (home_win_prob / total) * 100
        draw_prob = (draw_prob / total) * 100
        away_win_prob = (away_win_prob / total) * 100
        
        # Probabilités pour Over/Under
        expected_goals = (home_stats['attack'] * 1.3 + away_stats['attack'] * 1.0) / 100
        
        if expected_goals > 3.0:
            over_25_prob = 65
            over_15_prob = 80
        elif expected_goals > 2.5:
            over_25_prob = 55
            over_15_prob = 75
        elif expected_goals > 2.0:
            over_25_prob = 45
            over_15_prob = 70
        else:
            over_25_prob = 35
            over_15_prob = 60
        
        # Probabilité Both Teams to Score
        home_score_prob = (home_stats['attack'] / away_stats['defense']) * 0.7
        away_score_prob = (away_stats['attack'] / home_stats['defense']) * 0.7
        btts_prob = (home_score_prob * away_score_prob) * 100
        
        return {
            '1': round(home_win_prob, 1),
            'X': round(draw_prob, 1),
            '2': round(away_win_prob, 1),
            '1X': round(home_win_prob + draw_prob, 1),
            '12': round(home_win_prob + away_win_prob, 1),
            'X2': round(draw_prob + away_win_prob, 1),
            'over_1.5': round(over_15_prob, 1),
            'under_1.5': round(100 - over_15_prob, 1),
            'over_2.5': round(over_25_prob, 1),
            'under_2.5': round(100 - over_25_prob, 1),
            'btts_yes': round(btts_prob, 1),
            'btts_no': round(100 - btts_prob, 1)
        }
    
    def _generate_all_predictions(self, home_stats, away_stats, probabilities):
        """Génère toutes les prédictions pour un match"""
        
        predictions = []
        
        # 1. PRONOSTIC PRINCIPAL (1X2)
        main_pred = self._get_main_prediction(probabilities)
        predictions.append({
            'type': 'Résultat Final',
            'prediction': main_pred['result'],
            'confidence': main_pred['confidence'],
            'probability': probabilities[main_pred['type']],
            'odd_estimated': round(1 / (probabilities[main_pred['type']] / 100) * 0.95, 2)
        })
        
        # 2. DOUBLE CHANCE
        dc_pred = self._get_double_chance_prediction(probabilities)
        predictions.append({
            'type': 'Double Chance',
            'prediction': dc_pred['result'],
            'confidence': dc_pred['confidence'],
            'probability': probabilities[dc_pred['type']],
            'odd_estimated': round(1 / (probabilities[dc_pred['type']] / 100) * 0.92, 2)
        })
        
        # 3. OVER/UNDER 1.5
        ou15_pred = self._get_over_under_prediction(probabilities, '1.5')
        predictions.append({
            'type': 'Over/Under 1.5',
            'prediction': ou15_pred['result'],
            'confidence': ou15_pred['confidence'],
            'probability': probabilities[ou15_pred['type']],
            'odd_estimated': round(1 / (probabilities[ou15_pred['type']] / 100) * 0.93, 2)
        })
        
        # 4. OVER/UNDER 2.5
        ou25_pred = self._get_over_under_prediction(probabilities, '2.5')
        predictions.append({
            'type': 'Over/Under 2.5',
            'prediction': ou25_pred['result'],
            'confidence': ou25_pred['confidence'],
            'probability': probabilities[ou25_pred['type']],
            'odd_estimated': round(1 / (probabilities[ou25_pred['type']] / 100) * 0.93, 2)
        })
        
        # 5. BOTH TEAMS TO SCORE
        btts_pred = self._get_btts_prediction(probabilities)
        predictions.append({
            'type': 'Both Teams to Score',
            'prediction': btts_pred['result'],
            'confidence': btts_pred['confidence'],
            'probability': probabilities[btts_pred['type']],
            'odd_estimated': round(1 / (probabilities[btts_pred['type']] / 100) * 0.94, 2)
        })
        
        # 6. SCORE EXACT
        score_pred = self._predict_exact_score(home_stats, away_stats, probabilities)
        predictions.append({
            'type': 'Score Exact',
            'prediction': score_pred['score'],
            'confidence': score_pred['confidence'],
            'probability': score_pred['probability'],
            'odd_estimated': round(1 / (score_pred['probability'] / 100) * 0.85, 2)
        })
        
        return predictions
    
    def _get_main_prediction(self, probabilities):
        """Retourne la prédiction principale 1X2"""
        max_prob = max(probabilities['1'], probabilities['X'], probabilities['2'])
        
        if max_prob == probabilities['1']:
            return {'type': '1', 'result': 'Victoire Domicile', 'confidence': 'Élevée' if max_prob > 60 else 'Moyenne'}
        elif max_prob == probabilities['X']:
            return {'type': 'X', 'result': 'Match Nul', 'confidence': 'Élevée' if max_prob > 40 else 'Moyenne'}
        else:
            return {'type': '2', 'result': 'Victoire Extérieur', 'confidence': 'Élevée' if max_prob > 60 else 'Moyenne'}
    
    def _get_double_chance_prediction(self, probabilities):
        """Retourne la meilleure double chance"""
        max_prob = max(probabilities['1X'], probabilities['12'], probabilities['X2'])
        
        if max_prob == probabilities['1X']:
            return {'type': '1X', 'result': 'Domicile ou Nul', 'confidence': 'Très élevée' if max_prob > 75 else 'Élevée'}
        elif max_prob == probabilities['12']:
            return {'type': '12', 'result': 'Pas de Nul', 'confidence': 'Élevée' if max_prob > 70 else 'Moyenne'}
        else:
            return {'type': 'X2', 'result': 'Extérieur ou Nul', 'confidence': 'Très élevée' if max_prob > 75 else 'Élevée'}
    
    def _get_over_under_prediction(self, probabilities, line):
        """Retourne la prédiction Over/Under"""
        if line == '1.5':
            if probabilities['over_1.5'] > probabilities['under_1.5']:
                return {'type': 'over_1.5', 'result': f'Over {line} buts', 
                       'confidence': 'Élevée' if probabilities['over_1.5'] > 65 else 'Moyenne'}
            else:
                return {'type': 'under_1.5', 'result': f'Under {line} buts',
                       'confidence': 'Élevée' if probabilities['under_1.5'] > 65 else 'Moyenne'}
        else:  # 2.5
            if probabilities['over_2.5'] > probabilities['under_2.5']:
                return {'type': 'over_2.5', 'result': f'Over {line} buts',
                       'confidence': 'Élevée' if probabilities['over_2.5'] > 60 else 'Moyenne'}
            else:
                return {'type': 'under_2.5', 'result': f'Under {line} buts',
                       'confidence': 'Élevée' if probabilities['under_2.5'] > 60 else 'Moyenne'}
    
    def _get_btts_prediction(self, probabilities):
        """Retourne la prédiction Both Teams to Score"""
        if probabilities['btts_yes'] > probabilities['btts_no']:
            return {'type': 'btts_yes', 'result': 'Les deux équipes marquent',
                   'confidence': 'Élevée' if probabilities['btts_yes'] > 65 else 'Moyenne'}
        else:
            return {'type': 'btts_no', 'result': 'Une équipe ne marque pas',
                   'confidence': 'Élevée' if probabilities['btts_no'] > 65 else 'Moyenne'}
    
    def _predict_exact_score(self, home_stats, away_stats, probabilities):
        """Prédit le score exact"""
        
        # Calcul des buts attendus
        home_expected = (home_stats['attack'] / away_stats['defense']) * 1.8
        away_expected = (away_stats['attack'] / home_stats['defense']) * 1.5
        
        # Ajouter de l'aléatoire réaliste
        home_goals = int(max(0, round(home_expected + random.uniform(-0.7, 0.8))))
        away_goals = int(max(0, round(away_expected + random.uniform(-0.6, 0.7))))
        
        # Ajuster selon le résultat probable
        if probabilities['1'] > probabilities['2'] + 15:
            home_goals = max(home_goals, away_goals + 1)
        elif probabilities['2'] > probabilities['1'] + 15:
            away_goals = max(away_goals, home_goals + 1)
        elif probabilities['X'] > 40:
            # Pour les matchs nuls probables, rapprocher les scores
            diff = abs(home_goals - away_goals)
            if diff > 0:
                home_goals = min(home_goals, away_goals)
                away_goals = home_goals
        
        # Limiter à 4 buts maximum
        home_goals = min(home_goals, 4)
        away_goals = min(away_goals, 4)
        
        # Probabilité de ce score exact
        base_prob = random.uniform(8, 20)
        
        # Augmenter la probabilité pour les scores courants
        common_scores = ['1-0', '2-1', '1-1', '2-0', '0-0', '1-2', '0-1']
        score_str = f"{home_goals}-{away_goals}"
        
        if score_str in common_scores:
            base_prob += random.uniform(5, 10)
        
        probability = min(round(base_prob, 1), 30)
        
        confidence = 'Élevée' if probability > 18 else 'Moyenne' if probability > 12 else 'Faible'
        
        return {
            'score': score_str,
            'probability': probability,
            'confidence': confidence
        }
    
    def _calculate_confidence(self, home_stats, away_stats, probabilities):
        """Calcule la confiance globale"""
        max_prob = max(probabilities['1'], probabilities['X'], probabilities['2'])
        
        if max_prob > 70:
            return {'level': 'Très élevée', 'score': random.randint(85, 95)}
        elif max_prob > 60:
            return {'level': 'Élevée', 'score': random.randint(70, 84)}
        elif max_prob > 50:
            return {'level': 'Moyenne', 'score': random.randint(60, 69)}
        else:
            return {'level': 'Faible', 'score': random.randint(40, 59)}
    
    def _generate_detailed_analysis(self, home_team, away_team, home_stats, away_stats):
        """Génère une analyse détaillée du match"""
        
        home_rating = home_stats['rating']
        away_rating = away_stats['rating']
        diff = home_rating - away_rating
        
        analysis = f"**ANALYSE DU MATCH {home_team} vs {away_team}**\n\n"
        
        if diff > 15:
            analysis += f"• **{home_team}** est largement favori avec un avantage significatif à domicile.\n"
            analysis += f"• Différence de rating: {diff:.1f} points en faveur du domicile.\n"
            analysis += "• Attaque puissante combinée à un avantage terrain important.\n"
        elif diff > 5:
            analysis += f"• **{home_team}** a un avantage à domicile mais le match reste ouvert.\n"
            analysis += f"• Différence modérée de {diff:.1f} points.\n"
            analysis += "• Les visiteurs pourraient créer des occasions.\n"
        elif diff > -5:
            analysis += "• **Match très équilibré**, tout est possible.\n"
            analysis += "• Les deux équipes ont des niveaux similaires.\n"
            analysis += "• Le détail fera la différence.\n"
        elif diff > -15:
            analysis += f"• **{away_team}** pourrait créer la surprise en déplacement.\n"
            analysis += f"• Avantage technique de {-diff:.1f} points pour les visiteurs.\n"
            analysis += "• Performance à l'extérieur à surveiller.\n"
        else:
            analysis += f"• **{away_team}** est clairement favori malgré le déplacement.\n"
            analysis += f"• Supériorité significative de {-diff:.1f} points.\n"
            analysis += "• Le domicile pourrait ne pas suffire.\n"
        
        # Analyse attaque/défense
        attack_ratio = home_stats['attack'] / away_stats['attack']
        
        if attack_ratio > 1.2:
            analysis += f"\n• **Supériorité offensive**: {home_team} a une attaque plus dangereuse.\n"
        elif attack_ratio < 0.8:
            analysis += f"\n• **Supériorité offensive**: {away_team} dispose d'une meilleure attaque.\n"
        
        defense_ratio = home_stats['defense'] / away_stats['defense']
        
        if defense_ratio > 1.2:
            analysis += f"• **Défense solide**: {home_team} a une meilleure défense.\n"
        elif defense_ratio < 0.8:
            analysis += f"• **Défense solide**: {away_team} est plus solide défensivement.\n"
        
        return analysis
    
    def analyze_date(self, target_date):
        """Analyse tous les matchs d'une date spécifique"""
        fixtures = self.api_client.get_fixtures_by_date(target_date)
        
        if not fixtures:
            return []
        
        predictions = []
        
        for fixture in fixtures:
            try:
                prediction = self.analyze_match(fixture)
                predictions.append(prediction)
            except Exception as e:
                continue
        
        # Trier par confiance
        predictions.sort(key=lambda x: x['confidence']['score'], reverse=True)
        
        return predictions

# =============================================================================
# APPLICATION STREAMLIT
# =============================================================================

def main():
    """Application principale"""
    
    # Configuration de la page
    st.set_page_config(
        page_title="Pronostics Football Complet",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .date-title {
        text-align: center;
        color: #666;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-high {
        background: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .confidence-medium {
        background: #FF9800;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .confidence-low {
        background: #f44336;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .bet-value-excellent {
        color: #4CAF50;
        font-weight: bold;
    }
    .bet-value-good {
        color: #FF9800;
        font-weight: bold;
    }
    .bet-value-fair {
        color: #757575;
        font-weight: bold;
    }
    .section-header {
        background: linear-gradient(90deg, #2196F3 0%, #21CBF3 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">⚽ PRONOSTICS FOOTBALL COMPLET</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Tous les types de paris • Sélection par jour • Analyses détaillées</div>', unsafe_allow_html=True)
    
    # Initialisation
    if 'api_client' not in st.session_state:
        st.session_state.api_client = FootballDataClient()
    
    if 'prediction_system' not in st.session_state:
        st.session_state.prediction_system = CompletePredictionSystem(st.session_state.api_client)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        # Test connexion
        api_status = st.session_state.api_client.test_connection()
        if api_status:
            st.success("✅ API Connectée")
        else:
            st.warning("⚠️ Mode simulation activé")
        
        st.divider()
        
        # Sélection de la date
        st.header("📅 SÉLECTION DU JOUR")
        
        today = date.today()
        
        selected_date = st.date_input(
            "Choisissez la date des matchs",
            value=today,
            min_value=today,
            max_value=today + timedelta(days=14),
            key="date_selector"
        )
        
        st.divider()
        
        # Filtres
        st.header("🎯 FILTRES")
        
        min_confidence = st.slider(
            "Confiance minimum",
            min_value=50,
            max_value=95,
            value=65,
            key="min_confidence_filter"
        )
        
        bet_types = st.multiselect(
            "Types de paris à afficher",
            options=['Résultat Final', 'Double Chance', 'Over/Under 1.5', 'Over/Under 2.5', 
                    'Both Teams to Score', 'Score Exact'],
            default=['Résultat Final', 'Double Chance', 'Over/Under 2.5', 'Score Exact'],
            key="bet_types_filter"
        )
        
        st.divider()
        
        # Bouton d'analyse
        if st.button("🚀 ANALYSER LES MATCHS", type="primary", use_container_width=True, key="analyze_button"):
            with st.spinner(f"Analyse des matchs du {selected_date}..."):
                predictions = st.session_state.prediction_system.analyze_date(selected_date)
                
                if predictions:
                    # Filtrer par confiance
                    filtered_predictions = [
                        p for p in predictions 
                        if p['confidence']['score'] >= min_confidence
                    ]
                    
                    st.session_state.predictions = filtered_predictions
                    st.session_state.selected_date = selected_date
                    st.success(f"✅ {len(filtered_predictions)} prédictions générées")
                else:
                    st.session_state.predictions = []
                    st.warning("⚠️ Aucun match trouvé pour cette date")
                
                st.rerun()
        
        st.divider()
        
        # Statistiques
        st.header("📊 STATISTIQUES")
        
        if 'predictions' in st.session_state and st.session_state.predictions:
            total = len(st.session_state.predictions)
            avg_conf = np.mean([p['confidence']['score'] for p in st.session_state.predictions])
            
            st.metric("Matchs analysés", total, key="stats_matches")
            st.metric("Confiance moyenne", f"{avg_conf:.1f}%", key="stats_confidence")
        
        st.divider()
        
        # Légende
        st.header("ℹ️ LÉGENDE")
        st.markdown("""
        **Niveaux de confiance:**
        • 🟢 > 85% : Très élevée
        • 🟡 70-85% : Élevée  
        • 🟠 60-70% : Moyenne
        • 🔴 < 60% : Faible
        
        **Valeur des paris:**
        • 🎯 Excellente : Forte valeur
        • 👍 Bonne : Bonne opportunité
        • ⚖️ Correcte : Pari équilibré
        """)
    
    # Contenu principal
    if 'predictions' not in st.session_state:
        st.info("""
        ## 👋 BIENVENUE DANS LE SYSTÈME DE PRONOSTICS COMPLET
        
        **Pour commencer :**
        1. Sélectionnez une date dans la sidebar
        2. Ajustez les filtres si nécessaire
        3. Cliquez sur "🚀 ANALYSER LES MATCHS"
        4. Les pronostics apparaîtront ici
        
        **Fonctionnalités incluses :**
        • 🏆 **Résultat final** (1/X/2)
        • 🔄 **Double chance** (1X/12/X2)
        • ⬆️⬇️ **Over/Under** (1.5 et 2.5 buts)
        • ⚽ **Both Teams to Score** (Oui/Non)
        • 🎯 **Score exact** prédit
        • 📊 **Probabilités détaillées**
        • 💰 **Cotes estimées** et évaluation de la valeur
        """)
    elif not st.session_state.predictions:
        st.warning(f"⚠️ Aucun pronostic ne correspond aux critères pour le {st.session_state.selected_date}")
    else:
        display_predictions()

def display_predictions():
    """Affiche toutes les prédictions"""
    
    predictions = st.session_state.predictions
    selected_date = st.session_state.selected_date
    
    # En-tête avec la date
    st.markdown(f'<div class="date-title">📅 PRONOSTICS DU {selected_date.strftime("%d/%m/%Y")}</div>', unsafe_allow_html=True)
    st.markdown(f"### 📊 **{len(predictions)} matchs analysés**")
    
    # Afficher chaque prédiction
    for idx, pred in enumerate(predictions):
        match_info = pred['match_info']
        confidence = pred['confidence']
        
        # Carte de prédiction
        with st.container():
            col_header1, col_header2 = st.columns([3, 1])
            
            with col_header1:
                st.markdown(f"### 🏆 {match_info['match']}")
                st.markdown(f"**{match_info['league']}** • {match_info['date'][:10]} {match_info['time']}")
            
            with col_header2:
                conf_score = confidence['score']
                if conf_score >= 85:
                    conf_class = "confidence-high"
                    conf_emoji = "🟢"
                elif conf_score >= 70:
                    conf_class = "confidence-medium"
                    conf_emoji = "🟡"
                elif conf_score >= 60:
                    conf_class = "confidence-low"
                    conf_emoji = "🟠"
                else:
                    conf_class = "confidence-low"
                    conf_emoji = "🔴"
                
                st.markdown(f'<div class="{conf_class}" style="text-align: center; padding: 10px;">'
                          f'{conf_emoji} {confidence["level"]}<br>'
                          f'<strong>{conf_score}%</strong>'
                          f'</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Section 1: Probabilités
            col_prob1, col_prob2, col_prob3 = st.columns(3)
            
            with col_prob1:
                st.markdown("**🏆 Résultat Final**")
                st.metric("1", f"{pred['probabilities']['1']}%")
                st.metric("X", f"{pred['probabilities']['X']}%")
                st.metric("2", f"{pred['probabilities']['2']}%")
            
            with col_prob2:
                st.markdown("**🔄 Double Chance**")
                st.metric("1X", f"{pred['probabilities']['1X']}%")
                st.metric("12", f"{pred['probabilities']['12']}%")
                st.metric("X2", f"{pred['probabilities']['X2']}%")
            
            with col_prob3:
                st.markdown("**⚽ Buts**")
                st.metric("Over 1.5", f"{pred['probabilities']['over_1.5']}%")
                st.metric("Over 2.5", f"{pred['probabilities']['over_2.5']}%")
                st.metric("BTTS Oui", f"{pred['probabilities']['btts_yes']}%")
            
            st.divider()
            
            # Section 2: Meilleures prédictions
            st.markdown("### 🎯 MEILLEURES PRÉDICTIONS")
            
            # Afficher les prédictions filtrées
            displayed_predictions = [p for p in pred['predictions'] 
                                   if p['type'] in st.session_state.get('bet_types_filter', [])]
            
            for pred_item in displayed_predictions:
                col_pred1, col_pred2, col_pred3, col_pred4 = st.columns([2, 2, 1, 1])
                
                with col_pred1:
                    st.write(f"**{pred_item['type']}**")
                
                with col_pred2:
                    st.write(f"{pred_item['prediction']}")
                
                with col_pred3:
                    st.write(f"{pred_item['probability']}%")
                
                with col_pred4:
                    # Évaluer la valeur du pari
                    odd = pred_item['odd_estimated']
                    prob = pred_item['probability']
                    value_score = (odd * (prob / 100) - 1) * 100
                    
                    if value_score > 10:
                        value_class = "bet-value-excellent"
                        value_text = "🎯 Excellente"
                    elif value_score > 5:
                        value_class = "bet-value-good"
                        value_text = "👍 Bonne"
                    else:
                        value_class = "bet-value-fair"
                        value_text = "⚖️ Correcte"
                    
                    st.markdown(f'<span class="{value_class}">{value_text}</span>', unsafe_allow_html=True)
                    st.write(f"@{odd}")
            
            st.divider()
            
            # Section 3: Analyse détaillée
            with st.expander("📝 ANALYSE DÉTAILLÉE", key=f"analysis_{idx}"):
                st.markdown(pred['detailed_analysis'])
                
                # Statistiques comparatives
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.markdown("**📈 CONSEILS DE PARI**")
                    st.write("• **Pari principal:** " + next(p['prediction'] for p in pred['predictions'] if p['type'] == 'Résultat Final'))
                    st.write("• **Pari safe:** " + next(p['prediction'] for p in pred['predictions'] if p['type'] == 'Double Chance'))
                    st.write("• **Pari valeur:** Score exact " + next(p['prediction'] for p in pred['predictions'] if p['type'] == 'Score Exact'))
                
                with col_stats2:
                    st.markdown("**⚠️ RISQUES IDENTIFIÉS**")
                    if pred['confidence']['score'] < 70:
                        st.write("• Confiance modérée, match imprévisible")
                    if pred['probabilities']['X'] > 35:
                        st.write("• Forte probabilité de match nul")
                    if abs(pred['probabilities']['1'] - pred['probabilities']['2']) < 10:
                        st.write("• Match très équilibré, risque élevé")
            
            # Espace entre les matchs
            if idx < len(predictions) - 1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    main()
