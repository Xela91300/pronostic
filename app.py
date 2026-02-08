# app.py - Système Complet de Pronostics Football
# Version corrigée sans erreurs Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class FootballPredictor:
    """Système complet de prédiction football"""
    
    def __init__(self):
        # Base de données d'équipes réalistes
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
    
    def generate_fixtures_for_date(self, target_date):
        """Génère des matchs réalistes pour une date donnée"""
        all_teams = list(self.teams_database.keys())
        random.shuffle(all_teams)
        
        fixtures = []
        num_matches = random.randint(8, 12)
        
        for i in range(0, min(num_matches * 2, len(all_teams)), 2):
            if i + 1 >= len(all_teams):
                break
                
            home_team = all_teams[i]
            away_team = all_teams[i + 1]
            
            hour = random.randint(15, 22)
            minute = random.choice([0, 15, 30, 45])
            
            league = self._determine_league(home_team, away_team)
            
            fixtures.append({
                'date': f"{target_date.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00+00:00",
                'home_name': home_team,
                'away_name': away_team,
                'league_name': league['name'],
                'league_country': league['country']
            })
        
        return fixtures
    
    def _determine_league(self, team1, team2):
        """Détermine la ligue basée sur les équipes"""
        league_mapping = {
            'Ligue 1': ['PSG', 'Marseille', 'Lyon', 'Monaco', 'Lille', 'Nice', 'Rennes', 'Lens'],
            'Premier League': ['Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 
                              'Manchester United', 'Tottenham', 'Newcastle'],
            'La Liga': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia'],
            'Bundesliga': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen'],
            'Serie A': ['Inter Milan', 'AC Milan', 'Juventus', 'Napoli', 'Roma', 'Atalanta']
        }
        
        for league, teams in league_mapping.items():
            if team1 in teams and team2 in teams:
                return {'name': league, 'country': league.split()[0] if ' ' in league else league}
        
        league = random.choice(list(league_mapping.keys()))
        return {'name': league, 'country': league.split()[0] if ' ' in league else league}
    
    def analyze_match(self, fixture):
        """Analyse complète d'un match avec tous les types de paris"""
        
        home_team = fixture['home_name']
        away_team = fixture['away_name']
        
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)
        
        probabilities = self._calculate_probabilities(home_stats, away_stats)
        
        predictions = {
            'match_info': {
                'match': f"{home_team} vs {away_team}",
                'league': fixture.get('league_name', 'Unknown'),
                'date': fixture.get('date', ''),
                'time': fixture.get('date', '')[11:16] if len(fixture.get('date', '')) > 16 else ''
            },
            'probabilities': probabilities,
            'predictions': self._generate_all_predictions(home_stats, away_stats, probabilities),
            'confidence': self._calculate_confidence(probabilities),
            'analysis': self._generate_analysis(home_team, away_team, home_stats, away_stats)
        }
        
        return predictions
    
    def _calculate_probabilities(self, home_stats, away_stats):
        """Calcule les probabilités pour tous les résultats"""
        
        home_power = home_stats['rating'] * home_stats['home_power']
        away_power = away_stats['rating']
        
        total_power = home_power + away_power
        
        home_win_prob = (home_power / total_power) * 100 * 0.85
        away_win_prob = (away_power / total_power) * 100 * 0.85
        draw_prob = 100 - home_win_prob - away_win_prob
        
        attack_defense_ratio = (home_stats['attack'] / away_stats['defense']) / (away_stats['attack'] / home_stats['defense'])
        
        if attack_defense_ratio > 1.2:
            home_win_prob += 5
            draw_prob -= 3
            away_win_prob -= 2
        elif attack_defense_ratio < 0.8:
            away_win_prob += 5
            draw_prob -= 3
            home_win_prob -= 2
        
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob = (home_win_prob / total) * 100
        draw_prob = (draw_prob / total) * 100
        away_win_prob = (away_win_prob / total) * 100
        
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
        
        # 1. PRONOSTIC PRINCIPAL
        main_pred = self._get_main_prediction(probabilities)
        predictions.append({
            'type': 'Résultat Final',
            'prediction': main_pred['result'],
            'confidence': main_pred['confidence'],
            'probability': probabilities[main_pred['type']],
            'odd': round(1 / (probabilities[main_pred['type']] / 100) * 0.95, 2)
        })
        
        # 2. DOUBLE CHANCE
        dc_pred = self._get_double_chance_prediction(probabilities)
        predictions.append({
            'type': 'Double Chance',
            'prediction': dc_pred['result'],
            'confidence': dc_pred['confidence'],
            'probability': probabilities[dc_pred['type']],
            'odd': round(1 / (probabilities[dc_pred['type']] / 100) * 0.92, 2)
        })
        
        # 3. OVER/UNDER 1.5
        ou15_pred = self._get_over_under_prediction(probabilities, '1.5')
        predictions.append({
            'type': 'Over/Under 1.5',
            'prediction': ou15_pred['result'],
            'confidence': ou15_pred['confidence'],
            'probability': probabilities[ou15_pred['type']],
            'odd': round(1 / (probabilities[ou15_pred['type']] / 100) * 0.93, 2)
        })
        
        # 4. OVER/UNDER 2.5
        ou25_pred = self._get_over_under_prediction(probabilities, '2.5')
        predictions.append({
            'type': 'Over/Under 2.5',
            'prediction': ou25_pred['result'],
            'confidence': ou25_pred['confidence'],
            'probability': probabilities[ou25_pred['type']],
            'odd': round(1 / (probabilities[ou25_pred['type']] / 100) * 0.93, 2)
        })
        
        # 5. BOTH TEAMS TO SCORE
        btts_pred = self._get_btts_prediction(probabilities)
        predictions.append({
            'type': 'Both Teams to Score',
            'prediction': btts_pred['result'],
            'confidence': btts_pred['confidence'],
            'probability': probabilities[btts_pred['type']],
            'odd': round(1 / (probabilities[btts_pred['type']] / 100) * 0.94, 2)
        })
        
        # 6. SCORE EXACT
        score_pred = self._predict_exact_score(home_stats, away_stats, probabilities)
        predictions.append({
            'type': 'Score Exact',
            'prediction': score_pred['score'],
            'confidence': score_pred['confidence'],
            'probability': score_pred['probability'],
            'odd': round(1 / (score_pred['probability'] / 100) * 0.85, 2)
        })
        
        return predictions
    
    def _get_main_prediction(self, probabilities):
        max_prob = max(probabilities['1'], probabilities['X'], probabilities['2'])
        
        if max_prob == probabilities['1']:
            return {'type': '1', 'result': 'Victoire Domicile', 'confidence': 'Élevée' if max_prob > 60 else 'Moyenne'}
        elif max_prob == probabilities['X']:
            return {'type': 'X', 'result': 'Match Nul', 'confidence': 'Élevée' if max_prob > 40 else 'Moyenne'}
        else:
            return {'type': '2', 'result': 'Victoire Extérieur', 'confidence': 'Élevée' if max_prob > 60 else 'Moyenne'}
    
    def _get_double_chance_prediction(self, probabilities):
        max_prob = max(probabilities['1X'], probabilities['12'], probabilities['X2'])
        
        if max_prob == probabilities['1X']:
            return {'type': '1X', 'result': 'Domicile ou Nul', 'confidence': 'Très élevée' if max_prob > 75 else 'Élevée'}
        elif max_prob == probabilities['12']:
            return {'type': '12', 'result': 'Pas de Nul', 'confidence': 'Élevée' if max_prob > 70 else 'Moyenne'}
        else:
            return {'type': 'X2', 'result': 'Extérieur ou Nul', 'confidence': 'Très élevée' if max_prob > 75 else 'Élevée'}
    
    def _get_over_under_prediction(self, probabilities, line):
        if line == '1.5':
            if probabilities['over_1.5'] > probabilities['under_1.5']:
                return {'type': 'over_1.5', 'result': f'Over {line} buts', 
                       'confidence': 'Élevée' if probabilities['over_1.5'] > 65 else 'Moyenne'}
            else:
                return {'type': 'under_1.5', 'result': f'Under {line} buts',
                       'confidence': 'Élevée' if probabilities['under_1.5'] > 65 else 'Moyenne'}
        else:
            if probabilities['over_2.5'] > probabilities['under_2.5']:
                return {'type': 'over_2.5', 'result': f'Over {line} buts',
                       'confidence': 'Élevée' if probabilities['over_2.5'] > 60 else 'Moyenne'}
            else:
                return {'type': 'under_2.5', 'result': f'Under {line} buts',
                       'confidence': 'Élevée' if probabilities['under_2.5'] > 60 else 'Moyenne'}
    
    def _get_btts_prediction(self, probabilities):
        if probabilities['btts_yes'] > probabilities['btts_no']:
            return {'type': 'btts_yes', 'result': 'Les deux équipes marquent',
                   'confidence': 'Élevée' if probabilities['btts_yes'] > 65 else 'Moyenne'}
        else:
            return {'type': 'btts_no', 'result': 'Une équipe ne marque pas',
                   'confidence': 'Élevée' if probabilities['btts_no'] > 65 else 'Moyenne'}
    
    def _predict_exact_score(self, home_stats, away_stats, probabilities):
        home_expected = (home_stats['attack'] / away_stats['defense']) * 1.8
        away_expected = (away_stats['attack'] / home_stats['defense']) * 1.5
        
        home_goals = int(max(0, round(home_expected + random.uniform(-0.7, 0.8))))
        away_goals = int(max(0, round(away_expected + random.uniform(-0.6, 0.7))))
        
        if probabilities['1'] > probabilities['2'] + 15:
            home_goals = max(home_goals, away_goals + 1)
        elif probabilities['2'] > probabilities['1'] + 15:
            away_goals = max(away_goals, home_goals + 1)
        elif probabilities['X'] > 40:
            diff = abs(home_goals - away_goals)
            if diff > 0:
                home_goals = min(home_goals, away_goals)
                away_goals = home_goals
        
        home_goals = min(home_goals, 4)
        away_goals = min(away_goals, 4)
        
        base_prob = random.uniform(8, 20)
        score_str = f"{home_goals}-{away_goals}"
        
        common_scores = ['1-0', '2-1', '1-1', '2-0', '0-0', '1-2', '0-1']
        if score_str in common_scores:
            base_prob += random.uniform(5, 10)
        
        probability = min(round(base_prob, 1), 30)
        confidence = 'Élevée' if probability > 18 else 'Moyenne' if probability > 12 else 'Faible'
        
        return {'score': score_str, 'probability': probability, 'confidence': confidence}
    
    def _calculate_confidence(self, probabilities):
        max_prob = max(probabilities['1'], probabilities['X'], probabilities['2'])
        
        if max_prob > 70:
            return {'level': 'Très élevée', 'score': random.randint(85, 95)}
        elif max_prob > 60:
            return {'level': 'Élevée', 'score': random.randint(70, 84)}
        elif max_prob > 50:
            return {'level': 'Moyenne', 'score': random.randint(60, 69)}
        else:
            return {'level': 'Faible', 'score': random.randint(40, 59)}
    
    def _generate_analysis(self, home_team, away_team, home_stats, away_stats):
        home_rating = home_stats['rating']
        away_rating = away_stats['rating']
        diff = home_rating - away_rating
        
        analysis = f"**ANALYSE {home_team} vs {away_team}**\n\n"
        
        if diff > 15:
            analysis += f"• **{home_team}** largement favori à domicile\n"
            analysis += f"• Avantage de {diff:.1f} points\n"
        elif diff > 5:
            analysis += f"• **{home_team}** légèrement favori\n"
            analysis += f"• Différence de {diff:.1f} points\n"
        elif diff > -5:
            analysis += "• **Match très équilibré**\n"
            analysis += "• Niveaux similaires\n"
        elif diff > -15:
            analysis += f"• **{away_team}** pourrait surprendre\n"
            analysis += f"• Avantage de {-diff:.1f} points\n"
        else:
            analysis += f"• **{away_team}** clairement favori\n"
            analysis += f"• Supériorité de {-diff:.1f} points\n"
        
        return analysis

# =============================================================================
# APPLICATION STREAMLIT
# =============================================================================

def main():
    """Application principale"""
    
    # Configuration
    st.set_page_config(
        page_title="Pronostics Football Pro",
        page_icon="⚽",
        layout="wide"
    )
    
    # CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .match-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .confidence-high {
        background: #4CAF50;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .confidence-medium {
        background: #FF9800;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .confidence-low {
        background: #f44336;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .prediction-item {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">⚽ PRONOSTICS FOOTBALL PROFESSIONNELS</div>', unsafe_allow_html=True)
    st.markdown("### Tous les types de paris • Sélection par jour • Analyses détaillées")
    
    # Initialisation
    if 'predictor' not in st.session_state:
        st.session_state.predictor = FootballPredictor()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        st.info("Mode: Données simulées réalistes")
        
        st.divider()
        
        # Date sélection
        st.header("📅 CHOISIR LE JOUR")
        
        today = date.today()
        selected_date = st.date_input(
            "Date des matchs",
            value=today,
            min_value=today,
            max_value=today + timedelta(days=14)
        )
        
        st.divider()
        
        # Filtres
        st.header("🎯 FILTRES")
        
        min_confidence = st.slider(
            "Confiance minimum (%)",
            50, 95, 65
        )
        
        st.divider()
        
        # Bouton analyse
        if st.button("🚀 ANALYSER LES MATCHS", type="primary", use_container_width=True):
            with st.spinner(f"Analyse du {selected_date}..."):
                fixtures = st.session_state.predictor.generate_fixtures_for_date(selected_date)
                
                predictions = []
                for fixture in fixtures:
                    try:
                        prediction = st.session_state.predictor.analyze_match(fixture)
                        if prediction['confidence']['score'] >= min_confidence:
                            predictions.append(prediction)
                    except:
                        continue
                
                predictions.sort(key=lambda x: x['confidence']['score'], reverse=True)
                st.session_state.current_predictions = predictions
                st.session_state.selected_date = selected_date
                st.success(f"✅ {len(predictions)} prédictions générées")
        
        st.divider()
        
        # Stats
        st.header("📊 STATISTIQUES")
        
        if 'current_predictions' in st.session_state:
            predictions = st.session_state.current_predictions
            if predictions:
                total = len(predictions)
                avg_conf = np.mean([p['confidence']['score'] for p in predictions])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Matchs", total)
                with col2:
                    st.metric("Confiance", f"{avg_conf:.1f}%")
        
        st.divider()
        
        # Info
        st.header("ℹ️ GUIDE")
        st.markdown("""
        **Types de paris:**
        • 1/X/2 : Résultat final
        • 1X/12/X2 : Double chance
        • Over/Under : Nombre de buts
        • BTTS : Les deux marquent
        • Score exact : Score prédit
        """)
    
    # Contenu principal
    if 'current_predictions' not in st.session_state:
        show_welcome()
    elif not st.session_state.current_predictions:
        st.warning(f"Aucun pronostic pour le {st.session_state.selected_date}")
    else:
        show_predictions()

def show_welcome():
    """Page d'accueil"""
    st.markdown("""
    ## 👋 BIENVENUE
    
    **Système complet de pronostics football avec:**
    
    ### 🎯 TOUS LES TYPES DE PARIS
    1. **🏆 Résultat Final** (1/X/2)
    2. **🔄 Double Chance** (1X/12/X2)
    3. **⬆️ Over 1.5 buts**
    4. **⬇️ Under 1.5 buts**
    5. **⬆️ Over 2.5 buts**
    6. **⬇️ Under 2.5 buts**
    7. **⚽ Both Teams to Score**
    8. **🎯 Score Exact**
    
    ### 📊 ANALYSES DÉTAILLÉES
    • Probabilités calculées
    • Niveaux de confiance
    • Cotes estimées
    • Conseils personnalisés
    
    ### 📅 FONCTIONNEMENT
    1. **Choisissez une date** dans la sidebar
    2. **Ajustez les filtres** si besoin
    3. **Cliquez sur ANALYSER**
    4. **Consultez les pronostics**
    
    ---
    
    *Les données sont simulées de manière réaliste pour démontrer les fonctionnalités*
    """)

def show_predictions():
    """Affiche les prédictions"""
    predictions = st.session_state.current_predictions
    selected_date = st.session_state.selected_date
    
    st.markdown(f"## 📅 PRONOSTICS DU {selected_date.strftime('%d/%m/%Y')}")
    st.markdown(f"### 🏆 {len(predictions)} MATCHS ANALYSÉS")
    
    for idx, pred in enumerate(predictions):
        match_info = pred['match_info']
        confidence = pred['confidence']
        
        # Carte du match
        with st.container():
            st.markdown(f"### {match_info['match']}")
            st.markdown(f"**{match_info['league']}** • {match_info['date'][:10]} {match_info['time']}")
            
            # Confiance
            conf_score = confidence['score']
            if conf_score >= 85:
                conf_class = "confidence-high"
                conf_text = "TRÈS ÉLEVÉE"
            elif conf_score >= 70:
                conf_class = "confidence-medium"
                conf_text = "ÉLEVÉE"
            else:
                conf_class = "confidence-low"
                conf_text = "MOYENNE"
            
            st.markdown(f'<div class="{conf_class}" style="display: inline-block; padding: 8px 16px; margin: 10px 0;">{conf_text} ({conf_score}%)</div>', unsafe_allow_html=True)
            
            # Probabilités
            col1, col2, col3 = st.columns(3)
            probs = pred['probabilities']
            
            with col1:
                st.markdown("**Résultat Final**")
                st.metric("1", f"{probs['1']}%")
                st.metric("X", f"{probs['X']}%")
                st.metric("2", f"{probs['2']}%")
            
            with col2:
                st.markdown("**Double Chance**")
                st.metric("1X", f"{probs['1X']}%")
                st.metric("12", f"{probs['12']}%")
                st.metric("X2", f"{probs['X2']}%")
            
            with col3:
                st.markdown("**Buts**")
                st.metric("Over 2.5", f"{probs['over_2.5']}%")
                st.metric("BTTS Oui", f"{probs['btts_yes']}%")
                st.metric("Score Exact", f"{next(p['probability'] for p in pred['predictions'] if p['type'] == 'Score Exact')}%")
            
            # Meilleures prédictions
            st.markdown("### 🎯 MEILLEURES PRÉDICTIONS")
            
            for pred_item in pred['predictions']:
                if pred_item['type'] in ['Résultat Final', 'Double Chance', 'Over/Under 2.5', 'Score Exact']:
                    col_pred1, col_pred2, col_pred3, col_pred4 = st.columns([2, 2, 1, 1])
                    
                    with col_pred1:
                        st.write(f"**{pred_item['type']}**")
                    
                    with col_pred2:
                        st.write(pred_item['prediction'])
                    
                    with col_pred3:
                        st.write(f"{pred_item['probability']}%")
                    
                    with col_pred4:
                        # Évaluer la valeur
                        odd = pred_item['odd']
                        prob = pred_item['probability']
                        value_score = (odd * (prob / 100) - 1) * 100
                        
                        if value_score > 10:
                            value_emoji = "🎯"
                            value_text = "Excellente"
                        elif value_score > 5:
                            value_emoji = "👍"
                            value_text = "Bonne"
                        else:
                            value_emoji = "⚖️"
                            value_text = "Correcte"
                        
                        st.write(f"{value_emoji} {value_text}")
                        st.write(f"@{odd}")
            
            # Analyse
            with st.expander("📝 ANALYSE DÉTAILLÉE"):
                st.markdown(pred['analysis'])
                
                st.markdown("**📈 CONSEILS**")
                st.write("• **Pari principal:** " + next(p['prediction'] for p in pred['predictions'] if p['type'] == 'Résultat Final'))
                st.write("• **Pari sécurisé:** " + next(p['prediction'] for p in pred['predictions'] if p['type'] == 'Double Chance'))
                st.write("• **Pari valeur:** " + next(p['prediction'] for p in pred['predictions'] if p['type'] == 'Score Exact'))
            
            # Séparateur
            if idx < len(predictions) - 1:
                st.markdown("---")

# =============================================================================
# EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    main()
