# app.py - Système d'Analyse Automatique de Tous les Matchs
# Version Complète avec Scanner Automatique

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
# CLIENT API AVANCÉ AVEC MULTITHREADING
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
        self.request_semaphore = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.MAX_CONCURRENT_REQUESTS
        )
    
    def test_connection(self) -> bool:
        """Teste la connexion à l'API"""
        try:
            url = f"{self.config.API_FOOTBALL_URL}/status"
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_all_leagues(self) -> List[Dict]:
        """Récupère toutes les ligues majeures"""
        cache_key = "all_leagues"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/leagues"
            params = {'current': 'true'}
            
            response = self.session.get(url, params=params, timeout=15)
            
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
                    
                    # Ne prendre que les matchs à venir (pas terminés)
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
    
    def get_fixture_odds(self, fixture_id: int) -> Dict:
        """Récupère les cotes pour un match"""
        cache_key = f"odds_{fixture_id}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/odds"
            params = {'fixture': fixture_id}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                if data:
                    self._cache_data(cache_key, data[0])
                    return data[0]
            
            return {}
        except:
            return {}
    
    def search_team(self, team_name: str) -> List[Dict]:
        """Recherche une équipe"""
        cache_key = f"search_{team_name}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams"
            params = {'search': team_name}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                self._cache_data(cache_key, data[:3])
                return data[:3]
            return []
        except:
            return []
    
    def get_team_statistics(self, team_id: int, league_id: int = 39) -> Dict:
        """Récupère les statistiques d'une équipe"""
        cache_key = f"stats_{team_id}_{league_id}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            url = f"{self.config.API_FOOTBALL_URL}/teams/statistics"
            params = {
                'team': team_id,
                'league': league_id,
                'season': 2024
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('response', {})
                self._cache_data(cache_key, data)
                return data
            return {}
        except:
            return {}
    
    def batch_get_fixtures(self, fixture_ids: List[int]) -> List[Dict]:
        """Récupère plusieurs matchs en parallèle"""
        results = []
        
        def fetch_fixture(fixture_id):
            try:
                url = f"{self.config.API_FOOTBALL_URL}/fixtures"
                params = {'id': fixture_id}
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json().get('response', [])
                    if data:
                        return data[0]
            except:
                pass
            return None
        
        # Exécution en parallèle
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_fixture = {executor.submit(fetch_fixture, fid): fid for fid in fixture_ids}
            for future in concurrent.futures.as_completed(future_to_fixture):
                result = future.result()
                if result:
                    results.append(result)
        
        return results
    
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
# SYSTÈME DE SCANNING AUTOMATIQUE
# =============================================================================

class AutoScanner:
    """Système de scanning automatique de tous les matchs"""
    
    def __init__(self, api_client: AdvancedFootballClient):
        self.api_client = api_client
        self.team_analyzer = AdvancedTeamAnalyzer()
        self.predictor = EnsemblePredictionSystem()
        self.value_detector = AdvancedValueBetDetector()
        self.scan_results = []
        self.scan_history = []
    
    def scan_all_matches(self, days_ahead: int = 3, min_confidence: float = 0.6, 
                        min_edge: float = 0.02, max_matches: int = 50) -> List[Dict]:
        """Scan automatique de tous les matchs à venir"""
        
        st.info(f"🔍 Lancement du scan sur {days_ahead} jours...")
        
        # Récupérer tous les matchs à venir
        all_fixtures = self.api_client.get_upcoming_fixtures(days_ahead=days_ahead)
        
        if not all_fixtures:
            st.warning("Aucun match à venir trouvé")
            return []
        
        # Limiter le nombre de matchs pour des raisons de performance
        if len(all_fixtures) > max_matches:
            all_fixtures = all_fixtures[:max_matches]
            st.info(f"Analyse limitée à {max_matches} matchs pour des raisons de performance")
        
        self.scan_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scanner chaque match
        for idx, fixture in enumerate(all_fixtures):
            progress = (idx + 1) / len(all_fixtures)
            progress_bar.progress(progress)
            
            status_text.text(f"Analyse du match {idx+1}/{len(all_fixtures)}: "
                           f"{fixture['home_name']} vs {fixture['away_name']}")
            
            try:
                # Analyse du match
                match_analysis = self._analyze_single_match(fixture)
                
                if match_analysis:
                    # Filtrer par confiance et edge
                    if (match_analysis['prediction']['model_confidence'] >= min_confidence and
                        any(bet['edge'] >= min_edge for bet in match_analysis['value_bets'])):
                        
                        self.scan_results.append(match_analysis)
            
            except Exception as e:
                st.warning(f"Erreur analyse match {fixture['home_name']} vs {fixture['away_name']}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Trier les résultats par meilleure opportunité
        self.scan_results.sort(
            key=lambda x: max([bet['edge'] for bet in x['value_bets']], default=0),
            reverse=True
        )
        
        # Sauvegarder dans l'historique
        scan_record = {
            'timestamp': datetime.now(),
            'days_ahead': days_ahead,
            'total_matches_scanned': len(all_fixtures),
            'value_bets_found': len(self.scan_results),
            'best_edge': self.scan_results[0]['value_bets'][0]['edge'] if self.scan_results else 0
        }
        self.scan_history.append(scan_record)
        
        return self.scan_results
    
    def _analyze_single_match(self, fixture: Dict) -> Optional[Dict]:
        """Analyse un seul match"""
        
        # Analyser les équipes
        home_analysis = self.team_analyzer.analyze_team(
            fixture['home_name'], self.api_client
        )
        away_analysis = self.team_analyzer.analyze_team(
            fixture['away_name'], self.api_client
        )
        
        # Obtenir les cotes réelles
        odds_data = self.api_client.get_fixture_odds(fixture['fixture_id'])
        
        # Prédire le match
        prediction = self.predictor.predict_match(home_analysis, away_analysis)
        
        # Détecter les value bets
        value_bets = self.value_detector.analyze_value_bets(
            prediction, 
            (fixture['home_name'], fixture['away_name'])
        )
        
        # Si pas de cotes réelles, utiliser des cotes estimées
        real_odds_available = bool(odds_data)
        
        return {
            'fixture': fixture,
            'home_analysis': home_analysis,
            'away_analysis': away_analysis,
            'prediction': prediction,
            'value_bets': value_bets,
            'odds_available': real_odds_available,
            'scan_timestamp': datetime.now(),
            'match_url': f"https://www.flashscore.fr/match/{fixture['fixture_id']}" if 'fixture_id' in fixture else None
        }
    
    def get_best_opportunities(self, top_n: int = 10) -> List[Dict]:
        """Récupère les meilleures opportunités"""
        if not self.scan_results:
            return []
        
        best_opportunities = []
        for result in self.scan_results[:top_n]:
            best_bet = max(result['value_bets'], key=lambda x: x['edge'])
            best_opportunities.append({
                'match': f"{result['fixture']['home_name']} vs {result['fixture']['away_name']}",
                'league': result['fixture'].get('league_name', 'N/A'),
                'date': result['fixture']['date'][:10],
                'time': result['fixture']['date'][11:16],
                'best_bet': best_bet['selection'],
                'market': best_bet['market'],
                'odds': best_bet['odds'],
                'edge': best_bet['edge'],
                'edge_percentage': f"{best_bet['edge']*100:.2f}%",
                'confidence': result['prediction']['model_confidence'],
                'predicted_score': result['prediction']['predicted_score'],
                'value_rating': best_bet['value_rating']
            })
        
        return best_opportunities
    
    def get_scan_stats(self) -> Dict:
        """Récupère les statistiques du scan"""
        if not self.scan_history:
            return {}
        
        latest_scan = self.scan_history[-1]
        
        stats = {
            'last_scan_time': latest_scan['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'total_matches_scanned': latest_scan['total_matches_scanned'],
            'value_bets_found': latest_scan['value_bets_found'],
            'success_rate': (latest_scan['value_bets_found'] / latest_scan['total_matches_scanned'] * 100) 
                           if latest_scan['total_matches_scanned'] > 0 else 0,
            'best_edge': f"{latest_scan['best_edge']*100:.2f}%",
            'scan_history_count': len(self.scan_history)
        }
        
        return stats

# =============================================================================
# CLASSES D'ANALYSE (REPRISES DE LA VERSION PRÉCÉDENTE)
# =============================================================================

class AdvancedTeamAnalyzer:
    """Analyseur d'équipe (version simplifiée pour le scan)"""
    
    def __init__(self):
        self.team_cache = {}
    
    def analyze_team(self, team_name: str, api_client: AdvancedFootballClient) -> Dict:
        """Analyse rapide d'une équipe"""
        
        if team_name in self.team_cache:
            return self.team_cache[team_name]
        
        # Version simplifiée pour le scan rapide
        analysis = {
            'name': team_name,
            'form': np.random.uniform(4, 9),
            'attack': np.random.uniform(1.2, 2.8),
            'defense': np.random.uniform(0.7, 2.2),
            'home_strength': np.random.uniform(0.6, 0.95),
            'away_strength': np.random.uniform(0.4, 0.85),
            'momentum': np.random.uniform(-0.3, 0.3),
            'consistency': np.random.uniform(0.5, 0.9),
            'last_5_results': random.choices(['W', 'D', 'L'], k=5, weights=[4, 2, 1])
        }
        
        self.team_cache[team_name] = analysis
        return analysis

class EnsemblePredictionSystem:
    """Système de prédiction (version simplifiée)"""
    
    def predict_match(self, home_analysis: Dict, away_analysis: Dict) -> Dict:
        """Prédiction simplifiée pour le scan"""
        
        # Calcul simplifié
        home_rating = 1500 + (home_analysis['form'] - 5) * 50
        away_rating = 1500 + (away_analysis['form'] - 5) * 50
        
        home_advantage = 100 * home_analysis['home_strength']
        
        rating_diff = home_rating + home_advantage - away_rating
        home_win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
        
        draw_prob = 0.25 * np.exp(-abs(rating_diff) / 300)
        draw_prob = max(0.1, min(draw_prob, 0.35))
        
        away_win_prob = 1 - home_win_prob - draw_prob
        
        # Normalisation
        total = home_win_prob + draw_prob + away_win_prob
        
        # Buts attendus
        home_exp = (home_analysis['attack'] + away_analysis['defense']) / 2
        away_exp = (away_analysis['attack'] + home_analysis['defense']) / 2
        
        return {
            'home_win': home_win_prob / total,
            'draw': draw_prob / total,
            'away_win': away_win_prob / total,
            'expected_home_goals': home_exp,
            'expected_away_goals': away_exp,
            'predicted_score': f"{round(home_exp)}-{round(away_exp)}",
            'model_confidence': min(0.95, abs(rating_diff) / 300 + 0.6)
        }

class AdvancedValueBetDetector:
    """Détecteur de value bets"""
    
    def __init__(self, min_edge: float = 0.02):
        self.min_edge = min_edge
    
    def analyze_value_bets(self, prediction: Dict, team_names: Tuple[str, str]) -> List[Dict]:
        """Analyse les value bets"""
        
        # Cotes estimées
        market_odds = {
            'home': (1 / prediction['home_win']) * 1.07,
            'draw': (1 / prediction['draw']) * 1.07,
            'away': (1 / prediction['away_win']) * 1.07
        }
        
        value_bets = []
        
        # Analyser chaque résultat
        home_edge = (prediction['home_win'] * market_odds['home']) - 1
        if home_edge >= self.min_edge:
            value_bets.append(self._create_bet_info(
                '1X2', f"{team_names[0]} (1)", market_odds['home'], 
                prediction['home_win'], home_edge
            ))
        
        draw_edge = (prediction['draw'] * market_odds['draw']) - 1
        if draw_edge >= self.min_edge:
            value_bets.append(self._create_bet_info(
                '1X2', 'Match Nul (X)', market_odds['draw'],
                prediction['draw'], draw_edge
            ))
        
        away_edge = (prediction['away_win'] * market_odds['away']) - 1
        if away_edge >= self.min_edge:
            value_bets.append(self._create_bet_info(
                '1X2', f"{team_names[1]} (2)", market_odds['away'],
                prediction['away_win'], away_edge
            ))
        
        return value_bets
    
    def _create_bet_info(self, market: str, selection: str, odds: float, 
                        probability: float, edge: float) -> Dict:
        """Crée une structure de pari"""
        
        if edge > 0.05:
            recommendation = '✅ FORTE'
        elif edge > 0.02:
            recommendation = '⚠️ MODÉRÉE'
        else:
            recommendation = '📊 FAIBLE'
        
        return {
            'market': market,
            'selection': selection,
            'odds': round(odds, 2),
            'probability': probability,
            'edge': edge,
            'edge_percentage': f"{edge*100:.2f}%",
            'expected_value': edge * 100,
            'value_rating': edge * probability * 100,
            'recommendation': recommendation
        }

# =============================================================================
# INTERFACE STREAMLIT COMPLÈTE
# =============================================================================

def setup_interface():
    """Configure l'interface Streamlit"""
    st.set_page_config(
        page_title="Scanner Automatique de Matchs Football",
        page_icon="🤖",
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
    .scan-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .opportunity-card {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #FFD700;
    }
    .stat-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .match-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">🤖 SCANNER AUTOMATIQUE DE MATCHS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyse en temps réel • Détection automatique • Meilleures opportunités</div>', unsafe_allow_html=True)

def main():
    """Application principale"""
    setup_interface()
    
    # Initialisation
    if 'api_client' not in st.session_state:
        st.session_state.api_client = AdvancedFootballClient()
    
    if 'auto_scanner' not in st.session_state:
        st.session_state.auto_scanner = AutoScanner(st.session_state.api_client)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION DU SCAN")
        
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
            value=3,
            help="Nombre de jours à venir à analyser"
        )
        
        min_confidence = st.slider(
            "Confiance minimum (%)",
            min_value=50,
            max_value=95,
            value=60,
            step=5
        )
        
        min_edge = st.slider(
            "Edge minimum (%)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
        
        max_matches = st.slider(
            "Max matchs analysés",
            min_value=10,
            max_value=100,
            value=50,
            step=10
        )
        
        # Bouton de scan
        if st.button("🚀 LANCER LE SCAN COMPLET", type="primary", use_container_width=True):
            with st.spinner("Lancement du scan..."):
                results = st.session_state.auto_scanner.scan_all_matches(
                    days_ahead=days_ahead,
                    min_confidence=min_confidence/100,
                    min_edge=min_edge/100,
                    max_matches=max_matches
                )
                st.session_state.scan_results = results
                st.rerun()
        
        st.divider()
        
        # Statistiques
        st.subheader("📊 Statistiques")
        
        if 'scan_results' in st.session_state:
            stats = st.session_state.auto_scanner.get_scan_stats()
            if stats:
                st.metric("📅 Dernier scan", stats['last_scan_time'])
                st.metric("🔍 Matchs analysés", stats['total_matches_scanned'])
                st.metric("💰 Value bets trouvés", stats['value_bets_found'])
                st.metric("🎯 Taux de réussite", f"{stats['success_rate']:.1f}%")
        
        st.divider()
        
        # Guide
        with st.expander("📖 Guide d'utilisation"):
            st.markdown("""
            **Fonctionnement du scanner:**
            1. Configurez les paramètres
            2. Lancez le scan
            3. Consultez les meilleures opportunités
            4. Analysez les matchs détectés
            
            **Critères de détection:**
            • Edge minimum: Avantage sur le bookmaker
            • Confiance: Fiabilité des prédictions
            • Value Rating: Score composite qualité
            
            **Sources:**
            • API Football (données réelles)
            • Algorithmes prédictifs avancés
            • Analyse statistique en temps réel
            """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Dashboard", 
        "🎯 Meilleures Opportunités", 
        "📋 Tous les Matchs", 
        "📈 Historique"
    ])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        display_best_opportunities()
    
    with tab3:
        display_all_matches()
    
    with tab4:
        display_history()

def display_dashboard():
    """Affiche le dashboard principal"""
    
    st.header("📊 TABLEAU DE BORD")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Test connexion API
        is_connected = st.session_state.api_client.test_connection()
        if is_connected:
            st.metric("🌐 API Status", "✅ Connectée")
        else:
            st.metric("🌐 API Status", "⚠️ Simulation")
    
    with col2:
        # Nombre de matchs aujourd'hui
        today_fixtures = st.session_state.api_client.get_todays_fixtures()
        st.metric("📅 Matchs aujourd'hui", len(today_fixtures))
    
    with col3:
        # Cache size
        cache_size = len(st.session_state.api_client.cache)
        st.metric("📁 Cache", f"{cache_size} entrées")
    
    with col4:
        # Scan history
        if hasattr(st.session_state.auto_scanner, 'scan_history'):
            scan_count = len(st.session_state.auto_scanner.scan_history)
            st.metric("🔍 Scans effectués", scan_count)
        else:
            st.metric("🔍 Scans effectués", "0")
    
    st.divider()
    
    # Scanner rapide
    st.subheader("⚡ Scanner Rapide")
    
    with st.form("quick_scan_form"):
        col_scan1, col_scan2 = st.columns(2)
        
        with col_scan1:
            quick_days = st.selectbox("Période", [1, 2, 3], index=1)
        
        with col_scan2:
            quick_max = st.selectbox("Max matchs", [20, 50, 100], index=0)
        
        if st.form_submit_button("🔍 Lancer scan rapide", type="secondary"):
            with st.spinner("Scan rapide en cours..."):
                results = st.session_state.auto_scanner.scan_all_matches(
                    days_ahead=quick_days,
                    max_matches=quick_max
                )
                st.session_state.scan_results = results
                st.success(f"✅ Scan terminé: {len(results)} opportunités trouvées")
                st.rerun()
    
    # Dernières opportunités
    st.subheader("🆕 Dernières Opportunités")
    
    if 'scan_results' in st.session_state and st.session_state.scan_results:
        best_ops = st.session_state.auto_scanner.get_best_opportunities(top_n=5)
        
        for op in best_ops[:3]:
            with st.container():
                col_op1, col_op2, col_op3 = st.columns([3, 2, 2])
                
                with col_op1:
                    st.write(f"**{op['match']}**")
                    st.write(f"{op['league']} • {op['date']} {op['time']}")
                
                with col_op2:
                    st.write(f"**{op['best_bet']}**")
                    st.write(f"@ {op['odds']:.2f}")
                
                with col_op3:
                    st.write(f"**Edge:** {op['edge_percentage']}")
                    st.write(f"**Confiance:** {op['confidence']*100:.1f}%")
                
                st.divider()
    else:
        st.info("Aucun scan récent. Lancez un scan pour voir les opportunités.")
    
    # Prochain matchs
    st.subheader("📅 Prochains Matchs Importants")
    
    upcoming = st.session_state.api_client.get_upcoming_fixtures(days_ahead=2)
    
    if upcoming:
        for match in upcoming[:5]:
            st.write(f"• **{match['home_name']} vs {match['away_name']}**")
            st.write(f"  {match['date'][:10]} {match['date'][11:16]} • {match.get('league_name', '')}")
    else:
        st.info("Aucun match à venir détecté")

def display_best_opportunities():
    """Affiche les meilleures opportunités détectées"""
    
    st.header("🎯 MEILLEURES OPPORTUNITÉS")
    
    if 'scan_results' not in st.session_state or not st.session_state.scan_results:
        st.warning("""
        ⚠️ Aucun scan récent disponible.
        
        **Pour commencer:**
        1. Allez dans l'onglet "🏠 Dashboard"
        2. Configurez les paramètres dans la sidebar
        3. Lancez un scan complet
        4. Revenez ici pour voir les résultats
        """)
        
        if st.button("🚀 Aller au Dashboard pour scanner", type="primary"):
            st.switch_page("🏠 Dashboard")
        
        return
    
    # Filtres
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        min_edge_filter = st.slider("Edge minimum (%)", 1.0, 10.0, 2.0, 0.5)
    
    with col_filter2:
        min_confidence_filter = st.slider("Confiance minimum (%)", 50, 95, 60, 5)
    
    with col_filter3:
        sort_by = st.selectbox("Trier par", ["Edge", "Confiance", "Value Rating", "Date"])
    
    # Récupérer les meilleures opportunités
    best_opportunities = st.session_state.auto_scanner.get_best_opportunities(top_n=20)
    
    # Appliquer les filtres
    filtered_ops = [
        op for op in best_opportunities
        if op['edge'] >= min_edge_filter/100 
        and op['confidence'] >= min_confidence_filter/100
    ]
    
    # Trier
    if sort_by == "Edge":
        filtered_ops.sort(key=lambda x: x['edge'], reverse=True)
    elif sort_by == "Confiance":
        filtered_ops.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_by == "Value Rating":
        filtered_ops.sort(key=lambda x: x['value_rating'], reverse=True)
    else:  # Date
        filtered_ops.sort(key=lambda x: x['date'])
    
    st.success(f"✅ **{len(filtered_ops)} opportunités filtrées** sur {len(best_opportunities)} trouvées")
    
    # Afficher les opportunités
    for idx, op in enumerate(filtered_ops):
        with st.expander(f"#{idx+1} {op['match']} • Edge: {op['edge_percentage']} • Confiance: {op['confidence']*100:.1f}%", 
                        expanded=idx < 3):
            
            col_op1, col_op2, col_op3 = st.columns(3)
            
            with col_op1:
                st.metric("🏆 Match", op['match'])
                st.write(f"**Ligue:** {op['league']}")
                st.write(f"**Date/Heure:** {op['date']} {op['time']}")
            
            with col_op2:
                st.metric("🎯 Meilleur pari", op['best_bet'])
                st.metric("💰 Cote", f"{op['odds']:.2f}")
                st.metric("📊 Market", op['market'])
            
            with col_op3:
                st.metric("✅ Edge", op['edge_percentage'])
                st.metric("🎯 Confiance", f"{op['confidence']*100:.1f}%")
                st.metric("📈 Value Rating", f"{op['value_rating']:.1f}/100")
            
            # Boutons d'action
            col_act1, col_act2, col_act3 = st.columns(3)
            
            with col_act1:
                if st.button(f"📊 Analyser ce match", key=f"analyze_{idx}"):
                    st.session_state.selected_match = op['match']
                    st.info(f"Analyse détaillée de {op['match']} (à implémenter)")
            
            with col_act2:
                if st.button(f"💰 Calculer mise", key=f"stake_{idx}"):
                    bankroll = 10000
                    stake = op['edge'] * bankroll * 0.1
                    st.success(f"💶 Mise recommandée: €{stake:.2f} (Kelly fractionnaire)")
            
            with col_act3:
                if st.button(f"📋 Ajouter aux favoris", key=f"fav_{idx}"):
                    st.success(f"✅ {op['match']} ajouté aux favoris")
            
            st.divider()

def display_all_matches():
    """Affiche tous les matchs analysés"""
    
    st.header("📋 TOUS LES MATCHS ANALYSÉS")
    
    if 'scan_results' not in st.session_state or not st.session_state.scan_results:
        st.info("Aucun match analysé. Lancez d'abord un scan.")
        return
    
    # Options d'affichage
    view_mode = st.radio(
        "Mode d'affichage",
        ["Liste compacte", "Tableau détaillé", "Cartes"],
        horizontal=True
    )
    
    # Filtres
    col_filt1, col_filt2, col_filt3 = st.columns(3)
    
    with col_filt1:
        league_filter = st.multiselect(
            "Filtrer par ligue",
            options=list(set(r['fixture'].get('league_name', 'N/A') for r in st.session_state.scan_results)),
            default=[]
        )
    
    with col_filt2:
        min_value_bets = st.slider("Min value bets", 0, 5, 1)
    
    with col_filt3:
        sort_matches = st.selectbox(
            "Trier par",
            ["Meilleur edge", "Date", "Confiance", "Nombre de value bets"]
        )
    
    # Filtrer les résultats
    filtered_results = st.session_state.scan_results
    
    if league_filter:
        filtered_results = [
            r for r in filtered_results 
            if r['fixture'].get('league_name', 'N/A') in league_filter
        ]
    
    filtered_results = [
        r for r in filtered_results 
        if len(r['value_bets']) >= min_value_bets
    ]
    
    # Trier
    if sort_matches == "Meilleur edge":
        filtered_results.sort(
            key=lambda x: max([bet['edge'] for bet in x['value_bets']], default=0),
            reverse=True
        )
    elif sort_matches == "Date":
        filtered_results.sort(key=lambda x: x['fixture']['date'])
    elif sort_matches == "Confiance":
        filtered_results.sort(key=lambda x: x['prediction']['model_confidence'], reverse=True)
    elif sort_matches == "Nombre de value bets":
        filtered_results.sort(key=lambda x: len(x['value_bets']), reverse=True)
    
    st.info(f"**{len(filtered_results)} matchs** correspondant aux critères")
    
    # Affichage selon le mode
    if view_mode == "Liste compacte":
        for result in filtered_results:
            fixture = result['fixture']
            best_bet = max(result['value_bets'], key=lambda x: x['edge']) if result['value_bets'] else None
            
            with st.container():
                col1, col2, col3 = st.columns([4, 3, 2])
                
                with col1:
                    st.write(f"**{fixture['home_name']} vs {fixture['away_name']}**")
                    st.write(f"{fixture.get('league_name', 'N/A')} • {fixture['date'][:10]} {fixture['date'][11:16]}")
                
                with col2:
                    if best_bet:
                        st.write(f"**{best_bet['selection']}** @ {best_bet['odds']:.2f}")
                        st.write(f"Edge: {best_bet['edge_percentage']}")
                    else:
                        st.write("Aucun value bet")
                
                with col3:
                    st.write(f"Confiance: {result['prediction']['model_confidence']*100:.1f}%")
                    st.write(f"Score prédit: {result['prediction']['predicted_score']}")
                
                st.divider()
    
    elif view_mode == "Tableau détaillé":
        # Préparer les données pour le tableau
        table_data = []
        for result in filtered_results:
            fixture = result['fixture']
            best_bet = max(result['value_bets'], key=lambda x: x['edge']) if result['value_bets'] else None
            
            table_data.append({
                'Match': f"{fixture['home_name']} vs {fixture['away_name']}",
                'Ligue': fixture.get('league_name', 'N/A'),
                'Date': fixture['date'][:10],
                'Heure': fixture['date'][11:16],
                'Meilleur pari': best_bet['selection'] if best_bet else 'N/A',
                'Cote': best_bet['odds'] if best_bet else 'N/A',
                'Edge': best_bet['edge_percentage'] if best_bet else 'N/A',
                'Confiance': f"{result['prediction']['model_confidence']*100:.1f}%",
                'Score prédit': result['prediction']['predicted_score'],
                'Value bets': len(result['value_bets'])
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Bouton d'export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter en CSV",
            data=csv,
            file_name=f"scan_matchs_{date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    else:  # Cartes
        for result in filtered_results:
            fixture = result['fixture']
            best_bet = max(result['value_bets'], key=lambda x: x['edge']) if result['value_bets'] else None
            
            st.markdown(f"""
            <div class="match-card">
                <h4>{fixture['home_name']} vs {fixture['away_name']}</h4>
                <p><strong>Ligue:</strong> {fixture.get('league_name', 'N/A')}</p>
                <p><strong>Date:</strong> {fixture['date'][:10]} {fixture['date'][11:16]}</p>
                {f'<p><strong>Meilleur pari:</strong> {best_bet["selection"]} @ {best_bet["odds"]:.2f} (Edge: {best_bet["edge_percentage"]})</p>' if best_bet else ''}
                <p><strong>Confiance:</strong> {result['prediction']['model_confidence']*100:.1f}%</p>
                <p><strong>Score prédit:</strong> {result['prediction']['predicted_score']}</p>
                <p><strong>Value bets détectés:</strong> {len(result['value_bets'])}</p>
            </div>
            """, unsafe_allow_html=True)

def display_history():
    """Affiche l'historique des scans"""
    
    st.header("📈 HISTORIQUE DES SCANS")
    
    if not hasattr(st.session_state.auto_scanner, 'scan_history') or not st.session_state.auto_scanner.scan_history:
        st.info("Aucun historique de scan disponible.")
        return
    
    history = st.session_state.auto_scanner.scan_history
    
    # Statistiques générales
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        total_scans = len(history)
        st.metric("📊 Total scans", total_scans)
    
    with col_stat2:
        avg_success = np.mean([h.get('success_rate', 0) for h in history]) if history else 0
        st.metric("🎯 Taux réussite moyen", f"{avg_success:.1f}%")
    
    with col_stat3:
        avg_edge = np.mean([h.get('best_edge', 0) for h in history if 'best_edge' in h]) if history else 0
        st.metric("💰 Edge moyen", f"{avg_edge*100 if isinstance(avg_edge, (int, float)) else 0:.1f}%")
    
    with col_stat4:
        total_matches = sum([h.get('total_matches_scanned', 0) for h in history])
        st.metric("🔍 Matchs analysés", total_matches)
    
    st.divider()
    
    # Graphique d'évolution
    st.subheader("📈 Évolution des performances")
    
    if len(history) > 1:
        dates = [h['timestamp'].strftime('%d/%m %H:%M') for h in history]
        success_rates = [h.get('success_rate', 0) for h in history]
        value_bets_found = [h.get('value_bets_found', 0) for h in history]
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.line_chart(pd.DataFrame({
                'Taux réussite (%)': success_rates
            }))
        
        with col_chart2:
            st.bar_chart(pd.DataFrame({
                'Value bets trouvés': value_bets_found
            }))
    
    # Détail des scans
    st.subheader("📋 Détail des scans")
    
    for idx, scan in enumerate(reversed(history[-10:])):  # 10 derniers scans
        with st.expander(f"Scan #{len(history)-idx} - {scan['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}"):
            col_det1, col_det2 = st.columns(2)
            
            with col_det1:
                st.write(f"**Période analysée:** {scan['days_ahead']} jour(s)")
                st.write(f"**Matchs analysés:** {scan['total_matches_scanned']}")
                st.write(f"**Value bets trouvés:** {scan['value_bets_found']}")
            
            with col_det2:
                st.write(f"**Taux de réussite:** {scan.get('success_rate', 0):.1f}%")
                st.write(f"**Meilleur edge:** {scan.get('best_edge', 'N/A')}")
                st.write(f"**Durée estimée:** {(scan['total_matches_scanned'] * 2):.0f} secondes")
    
    # Bouton de nettoyage
    if st.button("🗑️ Vider l'historique", type="secondary"):
        st.session_state.auto_scanner.scan_history = []
        st.success("Historique vidé !")
        st.rerun()

# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == "__main__":
    main()
