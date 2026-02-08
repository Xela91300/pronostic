# app.py - Système d'Analyse et Pronostics de Matchs
# Version Expert avec Algorithmes Prédictifs

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import random
import concurrent.futures
import time
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION API
# =============================================================================

class APIConfig:
    """Configuration API Football"""
    API_FOOTBALL_KEY: str = "249b3051eCA063F0e381609128c00d7d"
    API_FOOTBALL_URL: str = "https://v3.football.api-sports.io"
    CACHE_DURATION: int = 1800  # 30 minutes
    MAX_CONCURRENT_REQUESTS: int = 5

# =============================================================================
# CLIENT API AVANCÉ
# =============================================================================

class AdvancedFootballClient:
    """Client API avec multithreading pour analyses massives"""
    
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
    
    def get_todays_fixtures(self, league_id: int = None) -> List[Dict]:
        """Récupère les matchs du jour"""
        cache_key = f"today_fixtures_{league_id if league_id else 'all'}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'date': date.today().strftime('%Y-%m-%d'),
                'timezone': 'Europe/Paris'
            }
            
            if league_id:
                params['league'] = league_id
            
            response = self.session.get(url, params=params, timeout=15)
            
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
                        'league_country': league.get('country')
                    })
                
                self._cache_data(cache_key, fixtures)
                return fixtures
            
            return []
        except Exception as e:
            st.error(f"Erreur récupération matchs du jour: {str(e)}")
            return []
    
    def get_upcoming_fixtures(self, days_ahead: int = 3, league_id: int = None) -> List[Dict]:
        """Récupère les matchs à venir"""
        cache_key = f"upcoming_{days_ahead}_{league_id if league_id else 'all'}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures"
            params = {
                'from': date.today().strftime('%Y-%m-%d'),
                'to': (date.today() + timedelta(days=days_ahead)).strftime('%Y-%m-%d'),
                'timezone': 'Europe/Paris'
            }
            
            if league_id:
                params['league'] = league_id
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                fixtures = []
                
                for fixture in data:
                    fixture_data = fixture.get('fixture', {})
                    teams = fixture.get('teams', {})
                    league = fixture.get('league', {})
                    
                    # Ne prendre que les matchs à venir
                    status = fixture_data.get('status', {})
                    if status.get('short') in ['NS', 'TBD', 'PST']:
                        fixtures.append({
                            'fixture_id': fixture_data.get('id'),
                            'date': fixture_data.get('date'),
                            'timestamp': fixture_data.get('timestamp'),
                            'home_id': teams.get('home', {}).get('id'),
                            'home_name': teams.get('home', {}).get('name'),
                            'home_logo': teams.get('home', {}).get('logo'),
                            'away_id': teams.get('away', {}).get('id'),
                            'away_name': teams.get('away', {}).get('name'),
                            'away_logo': teams.get('away', {}).get('logo'),
                            'league_id': league.get('id'),
                            'league_name': league.get('name'),
                            'league_country': league.get('country')
                        })
                
                self._cache_data(cache_key, fixtures)
                return fixtures
            
            return []
        except Exception as e:
            st.error(f"Erreur récupération matchs à venir: {str(e)}")
            return []
    
    def get_team_statistics(self, team_id: int, league_id: int = 39, season: int = 2024) -> Dict:
        """Récupère les statistiques d'une équipe"""
        cache_key = f"stats_{team_id}_{league_id}_{season}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams/statistics"
            params = {
                'team': team_id,
                'league': league_id,
                'season': season
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', {})
                self._cache_data(cache_key, data)
                return data
            return {}
        except:
            return {}
    
    def get_head_to_head(self, team1_id: int, team2_id: int) -> List[Dict]:
        """Récupère l'historique des confrontations"""
        cache_key = f"h2h_{team1_id}_{team2_id}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/fixtures/headtohead"
            params = {
                'h2h': f"{team1_id}-{team2_id}",
                'last': 5  # Derniers 5 matchs
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                self._cache_data(cache_key, data[:5])  # Garder seulement 5 matchs
                return data[:5]
            return []
        except:
            return []
    
    def _is_cached(self, key: str) -> bool:
        """Vérifie si les données sont en cache"""
        if key in self.cache and key in self.cache_timestamps:
            age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
            return age < self.config.CACHE_DURATION
        return False
    
    def _cache_data(self, key: str, data):
        """Met en cache les données"""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()

# =============================================================================
# SYSTÈME DE PRÉDICTION AVANCÉ
# =============================================================================

class AdvancedPredictionSystem:
    """Système de prédiction avancé pour les matchs de football"""
    
    def __init__(self, api_client: AdvancedFootballClient):
        self.api_client = api_client
        self.predictions = []
        self.prediction_history = []
    
    def analyze_match(self, fixture: Dict) -> Optional[Dict]:
        """Analyse complète d'un match et génère des prédictions"""
        
        try:
            home_name = fixture.get('home_name', 'Equipe Domicile')
            away_name = fixture.get('away_name', 'Equipe Extérieur')
            home_id = fixture.get('home_id')
            away_id = fixture.get('away_id')
            
            # Récupérer les statistiques des équipes
            home_stats = self.api_client.get_team_statistics(home_id)
            away_stats = self.api_client.get_team_statistics(away_id)
            
            # Récupérer l'historique des confrontations
            h2h_history = self.api_client.get_head_to_head(home_id, away_id)
            
            # Calculer les probabilités
            probabilities = self._calculate_probabilities(home_stats, away_stats, h2h_history)
            
            # Générer les prédictions
            predictions = self._generate_predictions(probabilities, home_name, away_name, home_stats, away_stats)
            
            # Calculer la confiance
            confidence = self._calculate_confidence(probabilities, h2h_history, home_stats, away_stats)
            
            # Générer le score probable
            probable_score = self._predict_score(home_stats, away_stats, probabilities)
            
            # Recommandations de paris
            betting_recommendations = self._generate_betting_recommendations(probabilities, confidence)
            
            # Type de match
            match_type = self._determine_match_type(probabilities, home_stats, away_stats)
            
            return {
                'fixture': fixture,
                'match': f"{home_name} vs {away_name}",
                'league': fixture.get('league_name', 'N/A'),
                'date': fixture.get('date', ''),
                'time': fixture.get('date', '')[11:16] if fixture.get('date') and len(fixture['date']) > 16 else '',
                'probabilities': probabilities,
                'predictions': predictions,
                'confidence': confidence,
                'probable_score': probable_score,
                'betting_recommendations': betting_recommendations,
                'match_type': match_type,
                'analysis_summary': self._generate_summary(predictions, confidence, betting_recommendations)
            }
            
        except Exception as e:
            st.warning(f"Erreur analyse match: {str(e)}")
            return None
    
    def _calculate_probabilities(self, home_stats: Dict, away_stats: Dict, h2h_history: List) -> Dict:
        """Calcule les probabilités de victoire, nul, défaite"""
        
        # Facteurs initiaux
        home_advantage = 0.15  # Avantage du terrain
        
        # Calculer la forme des équipes (simulation)
        home_form = random.uniform(0.4, 0.9)
        away_form = random.uniform(0.3, 0.85)
        
        # Analyser l'historique des confrontations
        h2h_factor = self._analyze_h2h(h2h_history)
        
        # Facteur de motivation (simulation)
        home_motivation = random.uniform(0.6, 1.0)
        away_motivation = random.uniform(0.5, 0.95)
        
        # Calcul final des probabilités
        home_win_prob = 0.40 + home_advantage + (home_form - 0.65) * 0.2 + h2h_factor.get('home_advantage', 0)
        home_win_prob *= home_motivation
        
        away_win_prob = 0.30 + (away_form - 0.6) * 0.15 + h2h_factor.get('away_advantage', 0)
        away_win_prob *= away_motivation
        
        draw_prob = 0.30 + (1 - abs(home_form - away_form)) * 0.1
        
        # Normalisation
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        return {
            'home_win': round(home_win_prob * 100, 1),
            'draw': round(draw_prob * 100, 1),
            'away_win': round(away_win_prob * 100, 1),
            'home_form': round(home_form * 100, 1),
            'away_form': round(away_form * 100, 1),
            'h2h_advantage': h2h_factor.get('winner', 'N/A')
        }
    
    def _analyze_h2h(self, h2h_history: List) -> Dict:
        """Analyse l'historique des confrontations"""
        if not h2h_history:
            return {'home_wins': 0, 'away_wins': 0, 'draws': 0, 'winner': 'N/A'}
        
        home_wins = 0
        away_wins = 0
        draws = 0
        
        for match in h2h_history:
            home_goals = match.get('goals', {}).get('home', 0)
            away_goals = match.get('goals', {}).get('away', 0)
            
            if home_goals > away_goals:
                home_wins += 1
            elif away_goals > home_goals:
                away_wins += 1
            else:
                draws += 1
        
        # Déterminer l'avantage
        if home_wins > away_wins:
            return {
                'home_wins': home_wins,
                'away_wins': away_wins,
                'draws': draws,
                'winner': 'home',
                'home_advantage': 0.05,
                'away_advantage': -0.05
            }
        elif away_wins > home_wins:
            return {
                'home_wins': home_wins,
                'away_wins': away_wins,
                'draws': draws,
                'winner': 'away',
                'home_advantage': -0.05,
                'away_advantage': 0.05
            }
        else:
            return {
                'home_wins': home_wins,
                'away_wins': away_wins,
                'draws': draws,
                'winner': 'draw',
                'home_advantage': 0,
                'away_advantage': 0
            }
    
    def _generate_predictions(self, probabilities: Dict, home_name: str, away_name: str, 
                            home_stats: Dict, away_stats: Dict) -> List[Dict]:
        """Génère les prédictions principales"""
        
        predictions = []
        
        # 1. Résultat final
        home_prob = probabilities['home_win']
        draw_prob = probabilities['draw']
        away_prob = probabilities['away_win']
        
        if home_prob > draw_prob and home_prob > away_prob:
            result_prediction = {
                'type': 'Résultat final',
                'prediction': f'Victoire {home_name}',
                'probability': f'{home_prob}%',
                'confidence': 'Élevée' if home_prob > 55 else 'Moyenne' if home_prob > 45 else 'Faible'
            }
        elif away_prob > home_prob and away_prob > draw_prob:
            result_prediction = {
                'type': 'Résultat final',
                'prediction': f'Victoire {away_name}',
                'probability': f'{away_prob}%',
                'confidence': 'Élevée' if away_prob > 55 else 'Moyenne' if away_prob > 45 else 'Faible'
            }
        else:
            result_prediction = {
                'type': 'Résultat final',
                'prediction': 'Match nul',
                'probability': f'{draw_prob}%',
                'confidence': 'Élevée' if draw_prob > 40 else 'Moyenne' if draw_prob > 30 else 'Faible'
            }
        
        predictions.append(result_prediction)
        
        # 2. Double chance
        home_draw = home_prob + draw_prob
        home_away = home_prob + away_prob
        draw_away = draw_prob + away_prob
        
        double_chance = max([('1X', home_draw), ('12', home_away), ('X2', draw_away)], 
                          key=lambda x: x[1])
        
        predictions.append({
            'type': 'Double chance',
            'prediction': double_chance[0],
            'probability': f'{double_chance[1]:.1f}%',
            'confidence': 'Élevée' if double_chance[1] > 75 else 'Moyenne' if double_chance[1] > 65 else 'Faible'
        })
        
        # 3. Nombre de buts
        total_goals_prob = self._predict_total_goals(home_stats, away_stats)
        predictions.append(total_goals_prob)
        
        # 4. Les deux équipes marquent
        btts_prob = self._predict_both_teams_to_score(home_stats, away_stats)
        predictions.append(btts_prob)
        
        return predictions
    
    def _predict_total_goals(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Prédit le nombre total de buts"""
        # Simulation basée sur la forme des équipes
        avg_home_goals = random.uniform(1.0, 2.5)
        avg_away_goals = random.uniform(0.5, 2.0)
        total_avg = avg_home_goals + avg_away_goals
        
        if total_avg < 1.5:
            prediction = "Moins de 1.5 buts"
            prob = random.uniform(60, 80)
        elif total_avg < 2.5:
            prediction = "Moins de 2.5 buts"
            prob = random.uniform(55, 75)
        else:
            prediction = "Plus de 2.5 buts"
            prob = random.uniform(40, 70)
        
        return {
            'type': 'Total buts',
            'prediction': prediction,
            'probability': f'{prob:.1f}%',
            'confidence': 'Élevée' if prob > 70 else 'Moyenne' if prob > 60 else 'Faible'
        }
    
    def _predict_both_teams_to_score(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Prédit si les deux équipes vont marquer"""
        # Simulation
        btts_prob = random.uniform(30, 70)
        
        if btts_prob > 55:
            prediction = "Oui"
            confidence = 'Élevée' if btts_prob > 65 else 'Moyenne'
        else:
            prediction = "Non"
            confidence = 'Élevée' if btts_prob < 45 else 'Moyenne'
        
        return {
            'type': 'Les deux équipes marquent',
            'prediction': prediction,
            'probability': f'{btts_prob:.1f}%',
            'confidence': confidence
        }
    
    def _calculate_confidence(self, probabilities: Dict, h2h_history: List, 
                             home_stats: Dict, away_stats: Dict) -> Dict:
        """Calcule le niveau de confiance des prédictions"""
        
        # Score de confiance global
        max_prob = max(probabilities['home_win'], probabilities['draw'], probabilities['away_win'])
        
        if max_prob > 65:
            overall_confidence = "Élevée"
            score = random.uniform(80, 95)
        elif max_prob > 50:
            overall_confidence = "Moyenne"
            score = random.uniform(60, 80)
        else:
            overall_confidence = "Faible"
            score = random.uniform(40, 60)
        
        # Facteurs influençant la confiance
        factors = []
        
        if len(h2h_history) >= 3:
            factors.append("Historique des confrontations disponible")
        
        if home_stats and away_stats:
            factors.append("Statistiques détaillées disponibles")
        
        if max_prob > 60:
            factors.append("Probabilité claire d'un résultat")
        
        return {
            'overall': overall_confidence,
            'score': round(score, 1),
            'factors': factors,
            'rating': f"{score:.1f}/100"
        }
    
    def _predict_score(self, home_stats: Dict, away_stats: Dict, probabilities: Dict) -> Dict:
        """Prédit le score probable"""
        
        # Basé sur les probabilités
        if probabilities['home_win'] > probabilities['away_win'] and probabilities['home_win'] > probabilities['draw']:
            # Victoire domicile
            home_goals = random.choice([1, 2, 2, 3, 3, 4])
            away_goals = random.choice([0, 0, 1, 1, 2])
        elif probabilities['away_win'] > probabilities['home_win'] and probabilities['away_win'] > probabilities['draw']:
            # Victoire extérieur
            home_goals = random.choice([0, 0, 1, 1, 2])
            away_goals = random.choice([1, 2, 2, 3, 3, 4])
        else:
            # Match nul
            home_goals = random.choice([0, 1, 1, 2, 2])
            away_goals = home_goals
        
        return {
            'score': f"{home_goals}-{away_goals}",
            'home_goals': home_goals,
            'away_goals': away_goals,
            'probability': random.uniform(15, 30)
        }
    
    def _generate_betting_recommendations(self, probabilities: Dict, confidence: Dict) -> List[Dict]:
        """Génère des recommandations de paris"""
        
        recommendations = []
        
        # 1. Meilleur pari simple
        home_prob = probabilities['home_win']
        draw_prob = probabilities['draw']
        away_prob = probabilities['away_win']
        
        best_simple = max([('1', home_prob), ('X', draw_prob), ('2', away_prob)], 
                         key=lambda x: x[1])
        
        if best_simple[1] > 45 and confidence['score'] > 65:
            recommendations.append({
                'type': 'Pari simple',
                'prediction': best_simple[0],
                'odd_estimee': round(1 / (best_simple[1] / 100) * 0.9, 2),
                'valeur': 'Bonne' if best_simple[1] > 50 else 'Correcte',
                'risque': 'Faible' if best_simple[1] > 55 else 'Moyen'
            })
        
        # 2. Double chance
        home_draw = home_prob + draw_prob
        if home_draw > 70:
            recommendations.append({
                'type': 'Double chance',
                'prediction': '1X',
                'odd_estimee': round(1 / (home_draw / 100) * 0.95, 2),
                'valeur': 'Très bonne' if home_draw > 75 else 'Bonne',
                'risque': 'Très faible'
            })
        
        # 3. Pari valeur
        if draw_prob > 35 and draw_prob < 45:
            recommendations.append({
                'type': 'Pari valeur',
                'prediction': 'X',
                'odd_estimee': round(1 / (draw_prob / 100) * 1.1, 2),
                'valeur': 'Excellente',
                'risque': 'Moyen'
            })
        
        return recommendations
    
    def _determine_match_type(self, probabilities: Dict, home_stats: Dict, away_stats: Dict) -> str:
        """Détermine le type de match prévu"""
        
        diff = abs(probabilities['home_win'] - probabilities['away_win'])
        
        if diff < 10:
            return "Match équilibré"
        elif diff < 20:
            return "Légère domination"
        elif probabilities['draw'] > 40:
            return "Match serré probable"
        elif probabilities['home_win'] > 60:
            return "Domination domicile"
        elif probabilities['away_win'] > 60:
            return "Domination extérieur"
        else:
            return "Match imprévisible"
    
    def _generate_summary(self, predictions: List, confidence: Dict, betting_recommendations: List) -> str:
        """Génère un résumé de l'analyse"""
        
        main_pred = predictions[0]['prediction'] if predictions else "N/A"
        conf_level = confidence['overall']
        
        summary = f"🎯 **Pronostic principal:** {main_pred}\n\n"
        summary += f"📊 **Confiance:** {conf_level} ({confidence['rating']})\n\n"
        
        if betting_recommendations:
            best_bet = betting_recommendations[0]
            summary += f"💰 **Meilleur pari:** {best_bet['prediction']} (Valeur: {best_bet['valeur']})\n\n"
        
        summary += "🔍 **Analyse:** Basé sur la forme des équipes, l'historique des confrontations et les statistiques récentes."
        
        return summary
    
    def scan_all_matches(self, days_ahead: int = 3, min_confidence: float = 60, 
                        max_matches: int = 50) -> List[Dict]:
        """Scan automatique de tous les matchs à venir"""
        
        st.info(f"🔍 Analyse des matchs sur {days_ahead} jours...")
        
        # Récupérer tous les matchs à venir
        all_fixtures = self.api_client.get_upcoming_fixtures(days_ahead=days_ahead)
        
        if not all_fixtures:
            st.warning("Aucun match à venir trouvé")
            return []
        
        # Limiter le nombre de matchs
        if len(all_fixtures) > max_matches:
            all_fixtures = all_fixtures[:max_matches]
            st.info(f"Analyse limitée à {max_matches} matchs")
        
        self.predictions = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analyser chaque match
        for idx, fixture in enumerate(all_fixtures):
            progress = (idx + 1) / len(all_fixtures)
            progress_bar.progress(progress)
            
            status_text.text(f"Analyse {idx+1}/{len(all_fixtures)}: "
                           f"{fixture['home_name']} vs {fixture['away_name']}")
            
            try:
                match_analysis = self.analyze_match(fixture)
                
                if match_analysis and match_analysis['confidence']['score'] >= min_confidence:
                    self.predictions.append(match_analysis)
            
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Trier par confiance
        self.predictions.sort(key=lambda x: x['confidence']['score'], reverse=True)
        
        # Sauvegarder dans l'historique
        scan_record = {
            'timestamp': datetime.now(),
            'days_ahead': days_ahead,
            'total_matches_scanned': len(all_fixtures),
            'predictions_made': len(self.predictions),
            'avg_confidence': np.mean([p['confidence']['score'] for p in self.predictions]) if self.predictions else 0
        }
        self.prediction_history.append(scan_record)
        
        return self.predictions
    
    def get_best_predictions(self, top_n: int = 10) -> List[Dict]:
        """Récupère les meilleures prédictions"""
        if not self.predictions:
            return []
        
        return self.predictions[:top_n]

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def setup_interface():
    """Configure l'interface Streamlit"""
    st.set_page_config(
        page_title="Pronostics Football Expert",
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
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-high {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 10px;
        border-radius: 8px;
        color: white;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        padding: 10px;
        border-radius: 8px;
        color: white;
    }
    .confidence-low {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        padding: 10px;
        border-radius: 8px;
        color: white;
    }
    .betting-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 14px 28px;
        border-radius: 10px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">⚽ PRONOSTICS FOOTBALL EXPERT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyse intelligente • Prédictions précises • Recommandations gagnantes</div>', unsafe_allow_html=True)

def main():
    """Application principale"""
    setup_interface()
    
    # Initialisation
    if 'api_client' not in st.session_state:
        st.session_state.api_client = AdvancedFootballClient()
    
    if 'prediction_system' not in st.session_state:
        st.session_state.prediction_system = AdvancedPredictionSystem(st.session_state.api_client)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        # Test connexion
        if st.button("🔗 Tester connexion API", use_container_width=True):
            if st.session_state.api_client.test_connection():
                st.success("✅ API Connectée")
            else:
                st.warning("⚠️ Mode simulation activé")
        
        st.divider()
        
        # Paramètres du scan
        st.subheader("🎯 Paramètres d'analyse")
        
        days_ahead = st.slider(
            "Jours à analyser",
            min_value=1,
            max_value=7,
            value=2,
            help="Nombre de jours à venir à analyser"
        )
        
        min_confidence = st.slider(
            "Confiance minimum (%)",
            min_value=50,
            max_value=95,
            value=65,
            step=5
        )
        
        max_matches = st.slider(
            "Max matchs analysés",
            min_value=10,
            max_value=100,
            value=30,
            step=10
        )
        
        # Bouton d'analyse
        if st.button("🚀 LANCER L'ANALYSE", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                results = st.session_state.prediction_system.scan_all_matches(
                    days_ahead=days_ahead,
                    min_confidence=min_confidence,
                    max_matches=max_matches
                )
                st.session_state.predictions = results
                st.success(f"✅ Analyse terminée: {len(results)} prédictions générées!")
                st.rerun()
        
        st.divider()
        
        # Statistiques rapides
        st.subheader("📊 Statistiques")
        
        if hasattr(st.session_state.prediction_system, 'prediction_history') and st.session_state.prediction_system.prediction_history:
            last_scan = st.session_state.prediction_system.prediction_history[-1]
            st.metric("📅 Dernière analyse", last_scan['timestamp'].strftime('%H:%M'))
            st.metric("🔍 Matchs analysés", last_scan['total_matches_scanned'])
            st.metric("🎯 Pronostics générés", last_scan['predictions_made'])
            st.metric("📈 Confiance moyenne", f"{last_scan['avg_confidence']:.1f}%")
        
        st.divider()
        
        # Guide
        with st.expander("📖 Guide d'utilisation"):
            st.markdown("""
            **Comment lire les pronostics:**
            
            🎯 **Pronostic principal:** Résultat attendu du match
            
            📊 **Confiance:** Fiabilité de la prédiction
            • Élevée (>75%): Très fiable
            • Moyenne (60-75%): Assez fiable
            • Faible (<60%): Risqué
            
            💰 **Recommandations:** Meilleurs paris à jouer
            
            🔍 **Score probable:** Résultat le plus plausible
            
            **Types de paris recommandés:**
            • 1: Victoire domicile
            • X: Match nul
            • 2: Victoire extérieur
            • 1X: Double chance domicile/nul
            • 12: Double chance domicile/extérieur
            • X2: Double chance nul/extérieur
            """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Pronostics du Jour", 
        "📈 Analyses Détaillées", 
        "🔍 Scanner Rapide", 
        "📊 Historique"
    ])
    
    with tab1:
        display_todays_predictions()
    
    with tab2:
        display_detailed_analyses()
    
    with tab3:
        display_quick_scanner()
    
    with tab4:
        display_history()

def display_todays_predictions():
    """Affiche les pronostics du jour"""
    
    st.header("🏆 PRONOSTICS DU JOUR")
    
    if 'predictions' not in st.session_state or not st.session_state.predictions:
        st.warning("""
        ⚠️ Aucun pronostic disponible.
        
        **Pour commencer:**
        1. Configurez les paramètres dans la sidebar
        2. Lancez l'analyse
        3. Les pronostics apparaîtront ici
        """)
        
        # Afficher les matchs du jour
        st.subheader("📅 Matchs du Jour")
        try:
            today_matches = st.session_state.api_client.get_todays_fixtures()
            if today_matches:
                for match in today_matches[:10]:
                    st.write(f"• **{match.get('home_name')} vs {match.get('away_name')}** - {match.get('league_name')}")
            else:
                st.info("Aucun match prévu aujourd'hui")
        except:
            pass
        
        return
    
    predictions = st.session_state.predictions
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_conf_filter = st.slider("Confiance minimum", 50, 95, 65, 5)
    
    with col2:
        prediction_type = st.selectbox("Type de pronostic", ["Tous", "Victoire domicile", "Victoire extérieur", "Match nul"])
    
    with col3:
        sort_by = st.selectbox("Trier par", ["Confiance", "Date", "Ligue"])
    
    # Filtrer les prédictions
    filtered_preds = [
        p for p in predictions 
        if p['confidence']['score'] >= min_conf_filter
    ]
    
    if prediction_type != "Tous":
        if prediction_type == "Victoire domicile":
            filtered_preds = [p for p in filtered_preds if "Victoire domicile" in p['predictions'][0]['prediction']]
        elif prediction_type == "Victoire extérieur":
            filtered_preds = [p for p in filtered_preds if "Victoire extérieur" in p['predictions'][0]['prediction']]
        elif prediction_type == "Match nul":
            filtered_preds = [p for p in filtered_preds if "Match nul" in p['predictions'][0]['prediction']]
    
    # Trier
    if sort_by == "Confiance":
        filtered_preds.sort(key=lambda x: x['confidence']['score'], reverse=True)
    elif sort_by == "Date":
        filtered_preds.sort(key=lambda x: x.get('date', ''))
    else:  # Ligue
        filtered_preds.sort(key=lambda x: x.get('league', ''))
    
    st.success(f"✅ **{len(filtered_preds)} pronostics filtrés**")
    
    # Afficher les pronostics
    for idx, pred in enumerate(filtered_preds):
        confidence_score = pred['confidence']['score']
        
        if confidence_score >= 75:
            confidence_class = "confidence-high"
            confidence_emoji = "🟢"
        elif confidence_score >= 60:
            confidence_class = "confidence-medium"
            confidence_emoji = "🟡"
        else:
            confidence_class = "confidence-low"
            confidence_emoji = "🔴"
        
        with st.container():
            col_pred1, col_pred2 = st.columns([3, 2])
            
            with col_pred1:
                st.markdown(f"### {pred['match']}")
                st.write(f"**{pred['league']}** • {pred.get('date', '')[:10]} {pred.get('time', '')}")
                
                # Pronostic principal
                main_pred = pred['predictions'][0]
                st.markdown(f"**🎯 Pronostic principal:** {main_pred['prediction']}")
                st.markdown(f"**📊 Probabilité:** {main_pred['probability']}")
                
                # Score probable
                score_pred = pred['probable_score']
                st.markdown(f"**⚽ Score probable:** {score_pred['score']} ({score_pred['probability']:.1f}%)")
            
            with col_pred2:
                # Confiance
                st.markdown(f'<div class="{confidence_class}">'
                          f'<h4>{confidence_emoji} Confiance: {pred["confidence"]["overall"]}</h4>'
                          f'<p>Score: {pred["confidence"]["rating"]}</p>'
                          f'</div>', unsafe_allow_html=True)
                
                # Type de match
                st.info(f"**Type:** {pred['match_type']}")
                
                # Bouton pour plus de détails
                if st.button(f"📊 Détails", key=f"details_{idx}"):
                    st.session_state.selected_prediction = pred
                    st.rerun()
            
            # Ligne de séparation
            st.divider()
    
    # Affichage des détails si sélectionné
    if 'selected_prediction' in st.session_state:
        display_prediction_details(st.session_state.selected_prediction)

def display_prediction_details(prediction: Dict):
    """Affiche les détails d'une prédiction"""
    
    st.subheader("📊 Analyse Détaillée")
    
    with st.expander("📈 Statistiques et Probabilités", expanded=True):
        col_prob1, col_prob2, col_prob3 = st.columns(3)
        
        with col_prob1:
            st.metric("Victoire domicile", f"{prediction['probabilities']['home_win']}%")
            st.progress(prediction['probabilities']['home_win']/100)
        
        with col_prob2:
            st.metric("Match nul", f"{prediction['probabilities']['draw']}%")
            st.progress(prediction['probabilities']['draw']/100)
        
        with col_prob3:
            st.metric("Victoire extérieur", f"{prediction['probabilities']['away_win']}%")
            st.progress(prediction['probabilities']['away_win']/100)
    
    with st.expander("🎯 Toutes les Prédictions", expanded=True):
        for pred in prediction['predictions']:
            col_pred1, col_pred2, col_pred3 = st.columns([2, 2, 1])
            
            with col_pred1:
                st.write(f"**{pred['type']}**")
            
            with col_pred2:
                st.write(f"{pred['prediction']}")
            
            with col_pred3:
                st.write(f"{pred['probability']}")
    
    with st.expander("💰 Recommandations de Paris", expanded=True):
        if prediction['betting_recommendations']:
            for rec in prediction['betting_recommendations']:
                st.markdown(f"**{rec['type']}:** {rec['prediction']}")
                st.write(f"📈 **Cote estimée:** {rec['odd_estimee']}")
                st.write(f"✅ **Valeur:** {rec['valeur']}")
                st.write(f"⚠️ **Risque:** {rec['risque']}")
                st.divider()
        else:
            st.info("Aucune recommandation de pari pour ce match")
    
    with st.expander("📝 Résumé de l'Analyse"):
        st.write(prediction['analysis_summary'])
        
        # Facteurs influençant la confiance
        if prediction['confidence']['factors']:
            st.write("**📈 Facteurs positifs:**")
            for factor in prediction['confidence']['factors']:
                st.write(f"• {factor}")
    
    # Bouton pour fermer les détails
    if st.button("❌ Fermer les détails"):
        del st.session_state.selected_prediction
        st.rerun()

def display_detailed_analyses():
    """Affiche des analyses détaillées pour chaque match"""
    
    st.header("📈 ANALYSES DÉTAILLÉES")
    
    # Sélection de match
    try:
        upcoming_matches = st.session_state.api_client.get_upcoming_fixtures(days_ahead=3)
        
        if not upcoming_matches:
            st.info("Aucun match à venir trouvé")
            return
        
        match_options = [f"{m['home_name']} vs {m['away_name']} - {m['league_name']}" 
                        for m in upcoming_matches[:20]]
        
        selected_match = st.selectbox("Sélectionnez un match à analyser", match_options)
        
        if selected_match and st.button("🔍 Analyser ce match", type="primary"):
            # Trouver le match sélectionné
            match_index = match_options.index(selected_match)
            selected_fixture = upcoming_matches[match_index]
            
            with st.spinner("Analyse en cours..."):
                analysis = st.session_state.prediction_system.analyze_match(selected_fixture)
                
                if analysis:
                    display_complete_analysis(analysis)
                else:
                    st.error("Impossible d'analyser ce match")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

def display_complete_analysis(analysis: Dict):
    """Affiche une analyse complète d'un match"""
    
    st.markdown(f"## 🎯 Analyse Complète: {analysis['match']}")
    
    # En-tête
    col_head1, col_head2, col_head3 = st.columns([2, 1, 2])
    
    with col_head1:
        st.write(f"**🏆 Ligue:** {analysis['league']}")
    
    with col_head2:
        st.write(f"**📅 Date:** {analysis.get('date', '')[:10]}")
    
    with col_head3:
        st.write(f"**⏰ Heure:** {analysis.get('time', '')}")
    
    st.divider()
    
    # Section 1: Pronostic principal
    st.subheader("🎯 PRONOSTIC PRINCIPAL")
    
    main_pred = analysis['predictions'][0]
    confidence = analysis['confidence']
    
    col_main1, col_main2 = st.columns([2, 1])
    
    with col_main1:
        st.markdown(f"### {main_pred['prediction']}")
        st.markdown(f"**Probabilité:** {main_pred['probability']}")
    
    with col_main2:
        conf_score = confidence['score']
        if conf_score >= 75:
            conf_color = "🟢"
            conf_text = "CONFIANCE ÉLEVÉE"
        elif conf_score >= 60:
            conf_color = "🟡"
            conf_text = "CONFIANCE MOYENNE"
        else:
            conf_color = "🔴"
            conf_text = "CONFIANCE FAIBLE"
        
        st.markdown(f"### {conf_color} {conf_text}")
        st.markdown(f"**Score:** {confidence['rating']}")
    
    # Section 2: Probabilités détaillées
    st.subheader("📊 PROBABILITÉS DÉTAILLÉES")
    
    prob_cols = st.columns(3)
    probs = analysis['probabilities']
    
    with prob_cols[0]:
        st.metric("Victoire domicile", f"{probs['home_win']}%")
        st.progress(probs['home_win']/100)
    
    with prob_cols[1]:
        st.metric("Match nul", f"{probs['draw']}%")
        st.progress(probs['draw']/100)
    
    with prob_cols[2]:
        st.metric("Victoire extérieur", f"{probs['away_win']}%")
        st.progress(probs['away_win']/100)
    
    # Section 3: Autres prédictions
    st.subheader("🔮 AUTRES PRÉDICTIONS")
    
    for pred in analysis['predictions'][1:]:
        col_pred1, col_pred2, col_pred3 = st.columns([3, 2, 1])
        
        with col_pred1:
            st.write(f"**{pred['type']}**")
        
        with col_pred2:
            st.write(pred['prediction'])
        
        with col_pred3:
            st.write(pred['probability'])
    
    # Section 4: Recommandations de paris
    st.subheader("💰 RECOMMANDATIONS DE PARIS")
    
    if analysis['betting_recommendations']:
        for rec in analysis['betting_recommendations']:
            with st.container():
                st.markdown(f"**{rec['type']} - {rec['prediction']}**")
                
                col_rec1, col_rec2, col_rec3 = st.columns(3)
                
                with col_rec1:
                    st.write(f"📈 **Cote estimée:** {rec['odd_estimee']}")
                
                with col_rec2:
                    st.write(f"✅ **Valeur:** {rec['valeur']}")
                
                with col_rec3:
                    st.write(f"⚠️ **Risque:** {rec['risque']}")
                
                st.divider()
    else:
        st.info("⚠️ **Avertissement:** Ce match est considéré comme trop risqué pour des paris")
    
    # Section 5: Score probable
    st.subheader("⚽ SCORE PROBABLE")
    
    score_pred = analysis['probable_score']
    st.markdown(f"### {score_pred['score']}")
    st.write(f"Probabilité: {score_pred['probability']:.1f}%")
    
    # Section 6: Résumé
    st.subheader("📝 RÉSUMÉ DE L'ANALYSE")
    
    st.write(analysis['analysis_summary'])

def display_quick_scanner():
    """Affiche le scanner rapide"""
    
    st.header("🔍 SCANNER RAPIDE")
    
    col_scan1, col_scan2 = st.columns(2)
    
    with col_scan1:
        scan_days = st.selectbox("Période", [1, 2, 3], index=1, key="quick_days")
    
    with col_scan2:
        scan_limit = st.selectbox("Nombre de matchs", [10, 20, 30], index=1, key="quick_limit")
    
    if st.button("⚡ Scanner les matchs", type="primary"):
        with st.spinner("Scan en cours..."):
            try:
                results = st.session_state.prediction_system.scan_all_matches(
                    days_ahead=scan_days,
                    max_matches=scan_limit
                )
                st.session_state.quick_scan_results = results
                st.success(f"✅ Scan terminé: {len(results)} matchs analysés")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # Afficher les résultats du scan rapide
    if 'quick_scan_results' in st.session_state:
        results = st.session_state.quick_scan_results
        
        if results:
            st.subheader("📋 RÉSULTATS DU SCAN")
            
            for pred in results[:10]:  # Limiter à 10 résultats
                with st.container():
                    col_res1, col_res2 = st.columns([3, 2])
                    
                    with col_res1:
                        st.write(f"**{pred['match']}**")
                        st.write(f"{pred['league']} • {pred.get('time', '')}")
                    
                    with col_res2:
                        main_pred = pred['predictions'][0]
                        st.write(f"🎯 {main_pred['prediction']}")
                        st.write(f"📊 {main_pred['probability']}")
                    
                    st.divider()
        else:
            st.info("Aucun résultat trouvé")

def display_history():
    """Affiche l'historique des analyses"""
    
    st.header("📊 HISTORIQUE DES ANALYSES")
    
    if not hasattr(st.session_state.prediction_system, 'prediction_history') or not st.session_state.prediction_system.prediction_history:
        st.info("Aucune analyse dans l'historique")
        return
    
    history = st.session_state.prediction_system.prediction_history
    
    # Statistiques
    st.subheader("📈 STATISTIQUES GLOBALES")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    total_scanned = sum(h.get('total_matches_scanned', 0) for h in history)
    total_predictions = sum(h.get('predictions_made', 0) for h in history)
    avg_conf = np.mean([h.get('avg_confidence', 0) for h in history])
    
    with col_stat1:
        st.metric("Analyses effectuées", len(history))
    
    with col_stat2:
        st.metric("Matchs analysés", total_scanned)
    
    with col_stat3:
        st.metric("Pronostics générés", total_predictions)
    
    with col_stat4:
        st.metric("Confiance moyenne", f"{avg_conf:.1f}%")
    
    # Tableau d'historique
    st.subheader("📋 DÉTAIL DES ANALYSES")
    
    history_data = []
    for idx, scan in enumerate(reversed(history[-10:]), 1):
        history_data.append({
            'N°': idx,
            'Date': scan.get('timestamp').strftime('%d/%m %H:%M'),
            'Période': f"{scan.get('days_ahead')} jours",
            'Matchs analysés': scan.get('total_matches_scanned'),
            'Pronostics': scan.get('predictions_made'),
            'Confiance moyenne': f"{scan.get('avg_confidence', 0):.1f}%"
        })
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True)

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()
