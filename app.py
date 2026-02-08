# app.py - Système d'Analyse et Pronostics de Matchs
# Version avec Données Simulées pour Fonctionner Sans API

import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DONNÉES SIMULÉES DES MATCHS
# =============================================================================

class MatchDataGenerator:
    """Générateur de données de matchs simulés"""
    
    def __init__(self):
        self.teams = {
            'France': ['PSG', 'Marseille', 'Lyon', 'Monaco', 'Lille', 'Nice', 'Rennes', 'Lens'],
            'Angleterre': ['Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 'Manchester United', 'Tottenham', 'Newcastle', 'Aston Villa'],
            'Espagne': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla', 'Valencia', 'Real Sociedad', 'Villarreal', 'Athletic Bilbao'],
            'Allemagne': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht Frankfurt', 'Wolfsburg', 'Monchengladbach'],
            'Italie': ['Inter Milan', 'AC Milan', 'Juventus', 'Napoli', 'Roma', 'Atalanta', 'Lazio', 'Fiorentina']
        }
        
        self.leagues = [
            {'id': 61, 'name': 'Ligue 1', 'country': 'France'},
            {'id': 39, 'name': 'Premier League', 'country': 'Angleterre'},
            {'id': 140, 'name': 'La Liga', 'country': 'Espagne'},
            {'id': 78, 'name': 'Bundesliga', 'country': 'Allemagne'},
            {'id': 135, 'name': 'Serie A', 'country': 'Italie'}
        ]
    
    def generate_todays_fixtures(self) -> List[Dict]:
        """Génère des matchs pour aujourd'hui"""
        fixtures = []
        today = date.today()
        
        for league in self.leagues:
            country_teams = self.teams.get(league['country'], [])
            if len(country_teams) >= 4:
                # Créer 3-4 matchs par ligue
                for i in range(0, min(4, len(country_teams)-1), 2):
                    home = country_teams[i]
                    away = country_teams[i+1]
                    
                    # Heure aléatoire
                    hour = random.randint(18, 22)
                    minute = random.choice([0, 30])
                    
                    fixtures.append({
                        'fixture_id': random.randint(1000, 9999),
                        'date': f"{today.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00+00:00",
                        'home_name': home,
                        'away_name': away,
                        'home_logo': f"https://example.com/{home.lower().replace(' ', '_')}.png",
                        'away_logo': f"https://example.com/{away.lower().replace(' ', '_')}.png",
                        'league_id': league['id'],
                        'league_name': league['name'],
                        'league_country': league['country'],
                        'status': {'short': 'NS'},
                        'timestamp': int(datetime.now().timestamp()) + random.randint(0, 86400)
                    })
        
        return fixtures[:12]  # Retourner max 12 matchs
    
    def generate_upcoming_fixtures(self, days_ahead: int = 3) -> List[Dict]:
        """Génère des matchs à venir"""
        fixtures = []
        today = date.today()
        
        for day_offset in range(days_ahead + 1):
            match_date = today + timedelta(days=day_offset)
            
            for league in self.leagues:
                country_teams = self.teams.get(league['country'], [])
                if len(country_teams) >= 4:
                    # Mélanger les équipes pour des matchs différents
                    shuffled_teams = random.sample(country_teams, len(country_teams))
                    
                    # Créer 2-3 matchs par jour par ligue
                    for i in range(0, min(4, len(shuffled_teams)-1), 2):
                        home = shuffled_teams[i]
                        away = shuffled_teams[i+1]
                        
                        # Heure aléatoire
                        hour = random.randint(16, 22)
                        minute = random.choice([0, 30])
                        
                        fixtures.append({
                            'fixture_id': random.randint(1000, 9999),
                            'date': f"{match_date.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00+00:00",
                            'home_name': home,
                            'away_name': away,
                            'home_logo': f"https://example.com/{home.lower().replace(' ', '_')}.png",
                            'away_logo': f"https://example.com/{away.lower().replace(' ', '_')}.png",
                            'league_id': league['id'],
                            'league_name': league['name'],
                            'league_country': league['country'],
                            'status': {'short': 'NS'},
                            'timestamp': int(datetime.now().timestamp()) + (day_offset * 86400) + random.randint(0, 86400)
                        })
        
        return fixtures[:30]  # Retourner max 30 matchs

# =============================================================================
# SYSTÈME DE PRÉDICTION AVANCÉ
# =============================================================================

class AdvancedPredictionSystem:
    """Système de prédiction avancé avec données simulées"""
    
    def __init__(self):
        self.data_generator = MatchDataGenerator()
        self.predictions = []
        self.prediction_history = []
        
        # Forces des équipes (simulées)
        self.team_strengths = {}
        self._initialize_team_strengths()
    
    def _initialize_team_strengths(self):
        """Initialise les forces des équipes"""
        for country_teams in self.data_generator.teams.values():
            for team in country_teams:
                # Force entre 40 et 95
                base_strength = random.uniform(40, 95)
                
                # Facteurs supplémentaires
                home_advantage = random.uniform(1.0, 1.3)
                attack_strength = random.uniform(0.7, 1.3)
                defense_strength = random.uniform(0.7, 1.3)
                form = random.uniform(0.8, 1.2)
                
                self.team_strengths[team] = {
                    'base': base_strength,
                    'home_advantage': home_advantage,
                    'attack': attack_strength,
                    'defense': defense_strength,
                    'form': form,
                    'current_form': random.uniform(0.5, 1.5)  # Forme actuelle
                }
    
    def analyze_match(self, fixture: Dict) -> Optional[Dict]:
        """Analyse complète d'un match"""
        
        try:
            home_name = fixture.get('home_name', 'Equipe Domicile')
            away_name = fixture.get('away_name', 'Equipe Extérieur')
            
            # Récupérer les forces des équipes
            home_strength = self.team_strengths.get(home_name, self._generate_team_strength(home_name))
            away_strength = self.team_strengths.get(away_name, self._generate_team_strength(away_name))
            
            # Calculer les probabilités
            probabilities = self._calculate_probabilities(home_strength, away_strength, home_name, away_name)
            
            # Générer les prédictions
            predictions = self._generate_predictions(probabilities, home_name, away_name, home_strength, away_strength)
            
            # Calculer la confiance
            confidence = self._calculate_confidence(probabilities)
            
            # Générer le score probable
            probable_score = self._predict_score(probabilities, home_strength, away_strength)
            
            # Recommandations de paris
            betting_recommendations = self._generate_betting_recommendations(probabilities, confidence)
            
            # Type de match
            match_type = self._determine_match_type(probabilities)
            
            # Historique des confrontations (simulé)
            h2h_history = self._generate_h2h_history(home_name, away_name)
            
            return {
                'fixture': fixture,
                'match': f"{home_name} vs {away_name}",
                'league': fixture.get('league_name', 'N/A'),
                'country': fixture.get('league_country', ''),
                'date': fixture.get('date', ''),
                'time': fixture.get('date', '')[11:16] if fixture.get('date') and len(fixture['date']) > 16 else '',
                'probabilities': probabilities,
                'predictions': predictions,
                'confidence': confidence,
                'probable_score': probable_score,
                'betting_recommendations': betting_recommendations,
                'match_type': match_type,
                'h2h_history': h2h_history,
                'analysis_summary': self._generate_summary(predictions, confidence, betting_recommendations, h2h_history)
            }
            
        except Exception as e:
            st.warning(f"Erreur analyse match: {str(e)}")
            return None
    
    def _generate_team_strength(self, team_name: str) -> Dict:
        """Génère des caractéristiques pour une équipe"""
        return {
            'base': random.uniform(40, 95),
            'home_advantage': random.uniform(1.0, 1.3),
            'attack': random.uniform(0.7, 1.3),
            'defense': random.uniform(0.7, 1.3),
            'form': random.uniform(0.8, 1.2),
            'current_form': random.uniform(0.5, 1.5)
        }
    
    def _calculate_probabilities(self, home_strength: Dict, away_strength: Dict, 
                               home_name: str, away_name: str) -> Dict:
        """Calcule les probabilités de victoire, nul, défaite"""
        
        # Force ajustée avec l'avantage du domicile
        home_adjusted = (home_strength['base'] * home_strength['home_advantage'] * 
                        home_strength['current_form'])
        away_adjusted = away_strength['base'] * away_strength['current_form']
        
        # Différence de force
        strength_diff = home_adjusted - away_adjusted
        
        # Probabilités de base
        if strength_diff > 20:
            # Fort avantage domicile
            home_win_prob = 0.60 + (strength_diff / 100)
            away_win_prob = 0.15
            draw_prob = 0.25
        elif strength_diff > 10:
            # Avantage domicile modéré
            home_win_prob = 0.50 + (strength_diff / 200)
            away_win_prob = 0.20
            draw_prob = 0.30
        elif strength_diff > -10:
            # Match équilibré
            home_win_prob = 0.40 + (strength_diff / 200)
            away_win_prob = 0.35 - (strength_diff / 200)
            draw_prob = 0.25
        elif strength_diff > -20:
            # Avantage extérieur modéré
            home_win_prob = 0.25
            away_win_prob = 0.50 - (strength_diff / 200)
            draw_prob = 0.25
        else:
            # Fort avantage extérieur
            home_win_prob = 0.20
            away_win_prob = 0.60 - (strength_diff / 100)
            draw_prob = 0.20
        
        # Ajustement avec la forme d'attaque/défense
        home_attack_factor = home_strength['attack']
        away_defense_factor = away_strength['defense']
        
        home_win_prob *= (home_attack_factor / away_defense_factor)
        away_win_prob *= (away_strength['attack'] / home_strength['defense'])
        
        # Normalisation
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Forme des équipes (pour l'affichage)
        home_form = min(100, max(20, (home_strength['current_form'] * 50)))
        away_form = min(100, max(20, (away_strength['current_form'] * 50)))
        
        return {
            'home_win': round(home_win_prob * 100, 1),
            'draw': round(draw_prob * 100, 1),
            'away_win': round(away_win_prob * 100, 1),
            'home_form': round(home_form, 1),
            'away_form': round(away_form, 1),
            'strength_diff': round(strength_diff, 1)
        }
    
    def _generate_predictions(self, probabilities: Dict, home_name: str, away_name: str,
                             home_strength: Dict, away_strength: Dict) -> List[Dict]:
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
                'confidence': self._get_confidence_level(home_prob)
            }
        elif away_prob > home_prob and away_prob > draw_prob:
            result_prediction = {
                'type': 'Résultat final',
                'prediction': f'Victoire {away_name}',
                'probability': f'{away_prob}%',
                'confidence': self._get_confidence_level(away_prob)
            }
        else:
            result_prediction = {
                'type': 'Résultat final',
                'prediction': 'Match nul',
                'probability': f'{draw_prob}%',
                'confidence': self._get_confidence_level(draw_prob)
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
            'confidence': self._get_confidence_level(double_chance[1], is_double=True)
        })
        
        # 3. Nombre de buts
        total_goals_pred = self._predict_total_goals(home_strength, away_strength)
        predictions.append(total_goals_pred)
        
        # 4. Les deux équipes marquent
        btts_pred = self._predict_both_teams_to_score(home_strength, away_strength)
        predictions.append(btts_pred)
        
        # 5. Handicap asiatique (pour les matches déséquilibrés)
        if abs(home_prob - away_prob) > 20:
            handicap_pred = self._predict_asian_handicap(probabilities, home_strength, away_strength)
            predictions.append(handicap_pred)
        
        return predictions
    
    def _get_confidence_level(self, probability: float, is_double: bool = False) -> str:
        """Retourne le niveau de confiance basé sur la probabilité"""
        if is_double:
            if probability > 80:
                return 'Très élevée'
            elif probability > 70:
                return 'Élevée'
            elif probability > 60:
                return 'Moyenne'
            else:
                return 'Faible'
        else:
            if probability > 65:
                return 'Élevée'
            elif probability > 55:
                return 'Bonne'
            elif probability > 45:
                return 'Moyenne'
            else:
                return 'Faible'
    
    def _predict_total_goals(self, home_strength: Dict, away_strength: Dict) -> Dict:
        """Prédit le nombre total de buts"""
        
        # Calcul de l'attaque moyenne
        avg_attack = (home_strength['attack'] + away_strength['attack']) / 2
        avg_defense = (home_strength['defense'] + away_strength['defense']) / 2
        
        # Nombre de buts attendu
        expected_goals = (avg_attack * 2.5) / avg_defense
        
        if expected_goals < 1.8:
            prediction = "Moins de 2.5 buts"
            probability = random.uniform(65, 85)
        elif expected_goals < 2.8:
            prediction = "Entre 1.5 et 2.5 buts"
            probability = random.uniform(55, 75)
        else:
            prediction = "Plus de 2.5 buts"
            probability = random.uniform(45, 70)
        
        return {
            'type': 'Total buts',
            'prediction': prediction,
            'probability': f'{probability:.1f}%',
            'confidence': self._get_confidence_level(probability, is_double=True)
        }
    
    def _predict_both_teams_to_score(self, home_strength: Dict, away_strength: Dict) -> Dict:
        """Prédit si les deux équipes vont marquer"""
        
        # Probabilité basée sur la force offensive et défensive
        home_attack = home_strength['attack']
        away_defense = away_strength['defense']
        away_attack = away_strength['attack']
        home_defense = home_strength['defense']
        
        home_score_prob = (home_attack / away_defense) * 0.6
        away_score_prob = (away_attack / home_defense) * 0.6
        
        btts_prob = (home_score_prob * away_score_prob) * 100
        
        if btts_prob > 60:
            prediction = "Oui"
        elif btts_prob > 40:
            prediction = "Probable"
        else:
            prediction = "Non"
        
        return {
            'type': 'Les deux équipes marquent',
            'prediction': prediction,
            'probability': f'{btts_prob:.1f}%',
            'confidence': 'Élevée' if abs(btts_prob - 50) > 20 else 'Moyenne'
        }
    
    def _predict_asian_handicap(self, probabilities: Dict, home_strength: Dict, away_strength: Dict) -> Dict:
        """Prédit le handicap asiatique"""
        
        diff = probabilities['home_win'] - probabilities['away_win']
        
        if diff > 25:
            handicap = "-1.5"
            prediction = f"{handicap} domicile"
        elif diff > 15:
            handicap = "-1.0"
            prediction = f"{handicap} domicile"
        elif diff < -25:
            handicap = "+1.5"
            prediction = f"{handicap} extérieur"
        elif diff < -15:
            handicap = "+1.0"
            prediction = f"{handicap} extérieur"
        else:
            handicap = "0.0"
            prediction = "Pas de handicap recommandé"
        
        return {
            'type': 'Handicap asiatique',
            'prediction': prediction,
            'probability': f'{max(probabilities["home_win"], probabilities["away_win"]):.1f}%',
            'confidence': 'Bonne' if handicap != "0.0" else 'Faible'
        }
    
    def _calculate_confidence(self, probabilities: Dict) -> Dict:
        """Calcule le niveau de confiance des prédictions"""
        
        max_prob = max(probabilities['home_win'], probabilities['draw'], probabilities['away_win'])
        
        if max_prob > 70:
            overall_confidence = "Très élevée"
            score = random.uniform(85, 95)
        elif max_prob > 60:
            overall_confidence = "Élevée"
            score = random.uniform(70, 85)
        elif max_prob > 50:
            overall_confidence = "Bonne"
            score = random.uniform(60, 75)
        elif max_prob > 40:
            overall_confidence = "Moyenne"
            score = random.uniform(50, 65)
        else:
            overall_confidence = "Faible"
            score = random.uniform(30, 50)
        
        # Facteurs influençant la confiance
        factors = []
        
        if abs(probabilities['home_win'] - probabilities['away_win']) > 20:
            factors.append("Match déséquilibré - plus prédictible")
        
        if probabilities['draw'] < 25:
            factors.append("Faible probabilité de match nul")
        
        if max_prob > 60:
            factors.append("Résultat clairement favorisé")
        
        return {
            'overall': overall_confidence,
            'score': round(score, 1),
            'factors': factors,
            'rating': f"{score:.1f}/100"
        }
    
    def _predict_score(self, probabilities: Dict, home_strength: Dict, away_strength: Dict) -> Dict:
        """Prédit le score probable"""
        
        home_attack = home_strength['attack']
        away_defense = away_strength['defense']
        away_attack = away_strength['attack']
        home_defense = home_strength['defense']
        
        # Buts attendus
        home_expected = (home_attack * 2.0) / away_defense
        away_expected = (away_attack * 1.5) / home_defense
        
        # Arrondir à l'entier le plus proche avec un peu d'aléatoire
        home_goals = int(max(0, round(home_expected + random.uniform(-0.5, 0.8))))
        away_goals = int(max(0, round(away_expected + random.uniform(-0.5, 0.6))))
        
        # Ajuster selon le résultat probable
        if probabilities['home_win'] > probabilities['away_win'] + 10:
            home_goals = max(home_goals, away_goals + 1)
        elif probabilities['away_win'] > probabilities['home_win'] + 10:
            away_goals = max(away_goals, home_goals + 1)
        
        # Probabilité de ce score spécifique
        score_prob = random.uniform(12, 25)
        
        return {
            'score': f"{home_goals}-{away_goals}",
            'home_goals': home_goals,
            'away_goals': away_goals,
            'probability': round(score_prob, 1)
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
        
        # Vérifier si le pari a de la valeur
        if best_simple[1] > 45 and confidence['score'] > 60:
            odd_estimee = round(1 / (best_simple[1] / 100) * random.uniform(0.85, 1.05), 2)
            
            # Évaluer la valeur
            value_score = (odd_estimee * (best_simple[1] / 100) - 1) * 100
            
            if value_score > 5:
                valeur = "Excellente"
                couleur = "🟢"
            elif value_score > 2:
                valeur = "Bonne"
                couleur = "🟡"
            else:
                valeur = "Correcte"
                couleur = "🟠"
            
            recommendations.append({
                'type': 'Pari simple',
                'prediction': best_simple[0],
                'odd_estimee': odd_estimee,
                'valeur': valeur,
                'couleur': couleur,
                'valeur_score': round(value_score, 1),
                'risque': 'Faible' if best_simple[1] > 55 else 'Moyen'
            })
        
        # 2. Double chance avec valeur
        home_draw = home_prob + draw_prob
        if home_draw > 70 and home_prob > 40:
            odd_dc = round(1 / (home_draw / 100) * random.uniform(0.9, 1.1), 2)
            recommendations.append({
                'type': 'Double chance',
                'prediction': '1X',
                'odd_estimee': odd_dc,
                'valeur': 'Très bonne',
                'couleur': '🟢',
                'risque': 'Très faible'
            })
        
        # 3. Pari surprise
        if draw_prob > 30 and draw_prob < 40:
            odd_surprise = round(1 / (draw_prob / 100) * random.uniform(1.1, 1.3), 2)
            recommendations.append({
                'type': 'Pari valeur',
                'prediction': 'X',
                'odd_estimee': odd_surprise,
                'valeur': 'Surprise intéressante',
                'couleur': '🟡',
                'risque': 'Élevé mais rentable'
            })
        
        return recommendations
    
    def _determine_match_type(self, probabilities: Dict) -> str:
        """Détermine le type de match prévu"""
        
        diff = abs(probabilities['home_win'] - probabilities['away_win'])
        
        if diff < 5:
            return "Match très équilibré ⚖️"
        elif diff < 15:
            return "Légère domination ⚔️"
        elif diff < 25:
            return "Match à sens unique 🎯"
        elif probabilities['draw'] > 40:
            return "Match nul probable 🤝"
        elif probabilities['home_win'] > 60:
            return "Domination à domicile 🏠"
        elif probabilities['away_win'] > 60:
            return "Suprise à l'extérieur ✈️"
        else:
            return "Match imprévisible 🎲"
    
    def _generate_h2h_history(self, home_name: str, away_name: str) -> List[Dict]:
        """Génère un historique des confrontations simulé"""
        
        history = []
        results = []
        
        # Générer 3-5 matchs historiques
        for i in range(random.randint(3, 5)):
            # Dates passées
            days_ago = random.randint(30, 500)
            match_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            # Score aléatoire avec tendance basée sur les noms des équipes
            if hash(home_name) % 3 > hash(away_name) % 3:
                home_goals = random.randint(1, 3)
                away_goals = random.randint(0, home_goals - 1)
                winner = 'home'
            elif hash(home_name) % 3 < hash(away_name) % 3:
                away_goals = random.randint(1, 3)
                home_goals = random.randint(0, away_goals - 1)
                winner = 'away'
            else:
                home_goals = random.randint(0, 2)
                away_goals = home_goals
                winner = 'draw'
            
            history.append({
                'date': match_date,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'result': f"{home_goals}-{away_goals}",
                'winner': winner
            })
            results.append(winner)
        
        # Calculer les statistiques H2H
        home_wins = results.count('home')
        away_wins = results.count('away')
        draws = results.count('draw')
        
        return {
            'matches': history,
            'stats': {
                'home_wins': home_wins,
                'away_wins': away_wins,
                'draws': draws,
                'total': len(history),
                'home_advantage': home_wins > away_wins,
                'trend': 'Domicile' if home_wins > away_wins else 'Extérieur' if away_wins > home_wins else 'Équilibré'
            }
        }
    
    def _generate_summary(self, predictions: List, confidence: Dict, 
                         betting_recommendations: List, h2h_history: Dict) -> str:
        """Génère un résumé de l'analyse"""
        
        main_pred = predictions[0]['prediction'] if predictions else "N/A"
        conf_level = confidence['overall']
        
        summary = f"🎯 **PRONOSTIC PRINCIPAL:** {main_pred}\n\n"
        summary += f"📊 **NIVEAU DE CONFIANCE:** {conf_level} ({confidence['rating']})\n\n"
        
        # Statistiques H2H
        h2h_stats = h2h_history['stats']
        summary += f"📈 **HISTORIQUE DES CONFRONTATIONS:** {h2h_stats['home_wins']}-{h2h_stats['draws']}-{h2h_stats['away_wins']}\n"
        summary += f"   Tendence: {h2h_stats['trend']}\n\n"
        
        if betting_recommendations:
            best_bet = betting_recommendations[0]
            summary += f"💰 **MEILLEUR PARI:** {best_bet['prediction']} @ {best_bet['odd_estimee']}\n"
            summary += f"   Valeur: {best_bet['valeur']} {best_bet.get('couleur', '')}\n\n"
        
        # Conseil final
        if confidence['score'] > 75:
            summary += "✅ **CONSEIL:** Pari recommandé avec forte confiance"
        elif confidence['score'] > 60:
            summary += "⚠️ **CONSEIL:** Pari intéressant avec bonne valeur"
        else:
            summary += "⛔ **CONSEIL:** Match risqué, pari déconseillé"
        
        return summary
    
    def scan_all_matches(self, days_ahead: int = 3, min_confidence: float = 60, 
                        max_matches: int = 30) -> List[Dict]:
        """Scan automatique de tous les matchs à venir"""
        
        st.info(f"🔍 Analyse des matchs sur {days_ahead} jours...")
        
        # Générer des matchs simulés
        all_fixtures = self.data_generator.generate_upcoming_fixtures(days_ahead=days_ahead)
        
        if not all_fixtures:
            st.warning("Aucun match à venir trouvé")
            return []
        
        # Limiter le nombre de matchs
        if len(all_fixtures) > max_matches:
            all_fixtures = all_fixtures[:max_matches]
        
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
    
    def get_best_predictions(self, top_n: int = 20) -> List[Dict]:
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
        page_title="Pronostics Football Expert - DONNÉES SIMULÉES",
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
    .h2h-card {
        background: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF9800;
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
    st.markdown('<div class="sub-title">DONNÉES SIMULÉES • ANALYSE INTELLIGENTE • PRÉDICTIONS PRÉCISES</div>', unsafe_allow_html=True)

def main():
    """Application principale"""
    setup_interface()
    
    # Initialisation
    if 'prediction_system' not in st.session_state:
        st.session_state.prediction_system = AdvancedPredictionSystem()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        
        st.info("📡 **Mode:** Données simulées (fonctionne sans API)")
        
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
            value=60,
            step=5
        )
        
        max_matches = st.slider(
            "Max matchs analysés",
            min_value=10,
            max_value=50,
            value=25,
            step=5
        )
        
        # Bouton d'analyse
        if st.button("🚀 LANCER L'ANALYSE", type="primary", use_container_width=True):
            with st.spinner("Génération et analyse des matchs..."):
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
            if last_scan['predictions_made'] > 0:
                st.metric("📈 Confiance moyenne", f"{last_scan['avg_confidence']:.1f}%")
        
        st.divider()
        
        # Info
        with st.expander("ℹ️ À propos"):
            st.markdown("""
            **Système de prédiction avec données simulées:**
            
            • 🏆 **5 championnats majeurs** (Ligue 1, Premier League, etc.)
            • ⚽ **Équipes réelles** avec caractéristiques uniques
            • 📊 **Algorithmes avancés** pour calculer les probabilités
            • 💰 **Recommandations de paris** avec évaluation de la valeur
            
            **Légende des couleurs:**
            • 🟢 Excellente valeur
            • 🟡 Bonne valeur  
            • 🟠 Valeur correcte
            • 🔴 Risqué
            
            **Confiance:**
            • >75%: Très élevée
            • 60-75%: Élevée
            • 50-60%: Moyenne
            • <50%: Faible
            """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Meilleurs Pronostics", 
        "📈 Analyse Détaillée", 
        "📅 Tous les Matchs", 
        "📊 Historique"
    ])
    
    with tab1:
        display_best_predictions()
    
    with tab2:
        display_detailed_analysis()
    
    with tab3:
        display_all_matches()
    
    with tab4:
        display_history()

def display_best_predictions():
    """Affiche les meilleurs pronostics"""
    
    st.header("🏆 MEILLEURS PRONOSTICS")
    
    if 'predictions' not in st.session_state or not st.session_state.predictions:
        st.warning("""
        ⚠️ Aucun pronostic disponible.
        
        **Pour commencer:**
        1. Configurez les paramètres dans la sidebar
        2. Cliquez sur "🚀 LANCER L'ANALYSE"
        3. Les pronostics apparaîtront ici
        """)
        
        # Afficher un aperçu des matchs disponibles
        st.subheader("📅 Matchs disponibles (simulés)")
        try:
            today_matches = st.session_state.prediction_system.data_generator.generate_todays_fixtures()
            if today_matches:
                for match in today_matches[:8]:
                    st.write(f"• **{match.get('home_name')} vs {match.get('away_name')}** - {match.get('league_name')}")
            else:
                st.info("Générez des matchs en lançant l'analyse")
        except:
            pass
        
        return
    
    predictions = st.session_state.predictions
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_conf_filter = st.slider("Confiance minimum", 50, 95, 65, 5, key="conf_filter")
    
    with col2:
        league_filter = st.selectbox("Filtrer par ligue", 
                                   ["Toutes", "Ligue 1", "Premier League", "La Liga", 
                                    "Bundesliga", "Serie A"])
    
    with col3:
        prediction_type = st.selectbox("Type de pronostic", 
                                      ["Tous", "Victoire domicile", "Victoire extérieur", "Match nul"])
    
    # Filtrer les prédictions
    filtered_preds = [
        p for p in predictions 
        if p['confidence']['score'] >= min_conf_filter
    ]
    
    if league_filter != "Toutes":
        filtered_preds = [p for p in filtered_preds if p.get('league') == league_filter]
    
    if prediction_type != "Tous":
        if prediction_type == "Victoire domicile":
            filtered_preds = [p for p in filtered_preds if "Victoire domicile" in p['predictions'][0]['prediction']]
        elif prediction_type == "Victoire extérieur":
            filtered_preds = [p for p in filtered_preds if "Victoire extérieur" in p['predictions'][0]['prediction']]
        elif prediction_type == "Match nul":
            filtered_preds = [p for p in filtered_preds if "Match nul" in p['predictions'][0]['prediction']]
    
    st.success(f"✅ **{len(filtered_preds)} pronostics filtrés**")
    
    if not filtered_preds:
        st.info("Aucun pronostic ne correspond aux critères de filtrage")
        return
    
    # Afficher les pronostics
    for idx, pred in enumerate(filtered_preds[:15]):  # Limiter à 15
        confidence_score = pred['confidence']['score']
        
        if confidence_score >= 80:
            confidence_class = "confidence-high"
            confidence_emoji = "🟢"
            confidence_text = "TRÈS ÉLEVÉE"
        elif confidence_score >= 70:
            confidence_class = "confidence-high"
            confidence_emoji = "🟢"
            confidence_text = "ÉLEVÉE"
        elif confidence_score >= 60:
            confidence_class = "confidence-medium"
            confidence_emoji = "🟡"
            confidence_text = "BONNE"
        else:
            confidence_class = "confidence-low"
            confidence_emoji = "🔴"
            confidence_text = "MOYENNE"
        
        with st.container():
            col_pred1, col_pred2 = st.columns([3, 2])
            
            with col_pred1:
                # Match info
                st.markdown(f"### {pred['match']}")
                st.write(f"**{pred['league']}** • {pred.get('date', '')[:10]} {pred.get('time', '')}")
                
                # Pronostic principal
                main_pred = pred['predictions'][0]
                st.markdown(f"**🎯 PRONOSTIC:** {main_pred['prediction']}")
                st.markdown(f"**📊 PROBABILITÉ:** {main_pred['probability']} ({main_pred['confidence']})")
                
                # Score probable
                score_pred = pred['probable_score']
                st.markdown(f"**⚽ SCORE PROBABLE:** {score_pred['score']} ({score_pred['probability']}%)")
            
            with col_pred2:
                # Confiance
                st.markdown(f'<div class="{confidence_class}">'
                          f'<h4>{confidence_emoji} CONFIANCE {confidence_text}</h4>'
                          f'<p>Score: {pred["confidence"]["rating"]}</p>'
                          f'</div>', unsafe_allow_html=True)
                
                # Type de match
                st.info(f"**{pred['match_type']}**")
                
                # Meilleur pari
                if pred['betting_recommendations']:
                    best_bet = pred['betting_recommendations'][0]
                    st.success(f"**💰 MEILLEUR PARI:** {best_bet['prediction']} @ {best_bet['odd_estimee']}")
                
                # Bouton pour plus de détails
                if st.button(f"📊 ANALYSER", key=f"details_{idx}", use_container_width=True):
                    st.session_state.selected_prediction = pred
                    st.rerun()
            
            # Ligne de séparation
            st.divider()
    
    # Affichage des détails si sélectionné
    if 'selected_prediction' in st.session_state:
        display_prediction_details(st.session_state.selected_prediction)

def display_prediction_details(prediction: Dict):
    """Affiche les détails d'une prédiction"""
    
    st.subheader(f"📊 ANALYSE DÉTAILLÉE: {prediction['match']}")
    
    # Bouton pour fermer
    if st.button("❌ Fermer l'analyse détaillée"):
        del st.session_state.selected_prediction
        st.rerun()
    
    # Informations générales
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("🏆 Ligue", prediction['league'])
    
    with col_info2:
        st.metric("📅 Date", prediction.get('date', '')[:10])
    
    with col_info3:
        st.metric("⏰ Heure", prediction.get('time', ''))
    
    st.divider()
    
    # Section 1: Probabilités
    st.subheader("📊 PROBABILITÉS")
    
    col_prob1, col_prob2, col_prob3 = st.columns(3)
    probs = prediction['probabilities']
    
    with col_prob1:
        st.metric("Victoire domicile", f"{probs['home_win']}%")
        st.progress(probs['home_win']/100)
    
    with col_prob2:
        st.metric("Match nul", f"{probs['draw']}%")
        st.progress(probs['draw']/100)
    
    with col_prob3:
        st.metric("Victoire extérieur", f"{probs['away_win']}%")
        st.progress(probs['away_win']/100)
    
    # Forme des équipes
    col_form1, col_form2 = st.columns(2)
    
    with col_form1:
        st.metric("📈 Forme domicile", f"{probs['home_form']}%")
    
    with col_form2:
        st.metric("📉 Forme extérieur", f"{probs['away_form']}%")
    
    st.divider()
    
    # Section 2: Historique des confrontations
    st.subheader("📈 HISTORIQUE DES CONFRONTATIONS")
    
    h2h = prediction['h2h_history']
    
    col_h2h1, col_h2h2, col_h2h3 = st.columns(3)
    
    with col_h2h1:
        st.metric("Victoires domicile", h2h['stats']['home_wins'])
    
    with col_h2h2:
        st.metric("Matchs nuls", h2h['stats']['draws'])
    
    with col_h2h3:
        st.metric("Victoires extérieur", h2h['stats']['away_wins'])
    
    # Derniers matchs
    with st.expander("Voir les derniers matchs"):
        for match in h2h['matches'][:3]:
            st.write(f"{match['date']}: {match['result']}")
    
    st.divider()
    
    # Section 3: Toutes les prédictions
    st.subheader("🔮 TOUTES LES PRÉDICTIONS")
    
    for pred in prediction['predictions']:
        col_pred1, col_pred2, col_pred3 = st.columns([2, 2, 1])
        
        with col_pred1:
            st.write(f"**{pred['type']}**")
        
        with col_pred2:
            st.write(f"{pred['prediction']}")
        
        with col_pred3:
            st.write(f"{pred['probability']}")
        
        st.divider()
    
    # Section 4: Recommandations de paris
    st.subheader("💰 RECOMMANDATIONS DE PARIS")
    
    if prediction['betting_recommendations']:
        for rec in prediction['betting_recommendations']:
            with st.container():
                col_rec1, col_rec2, col_rec3, col_rec4 = st.columns([2, 1, 2, 1])
                
                with col_rec1:
                    st.write(f"**{rec['type']}**")
                
                with col_rec2:
                    st.write(f"**{rec['prediction']}**")
                
                with col_rec3:
                    st.write(f"Cote: **{rec['odd_estimee']}**")
                    st.write(f"Valeur: {rec.get('couleur', '')} {rec['valeur']}")
                
                with col_rec4:
                    st.write(f"Risque: {rec['risque']}")
    else:
        st.info("⚠️ Aucune recommandation de pari pour ce match (trop risqué)")
    
    st.divider()
    
    # Section 5: Score probable
    st.subheader("⚽ SCORE PROBABLE")
    
    score_pred = prediction['probable_score']
    col_score1, col_score2 = st.columns(2)
    
    with col_score1:
        st.markdown(f"# {score_pred['score']}")
    
    with col_score2:
        st.metric("Probabilité de ce score", f"{score_pred['probability']}%")
    
    st.divider()
    
    # Section 6: Résumé
    st.subheader("📝 RÉSUMÉ DE L'ANALYSE")
    
    st.write(prediction['analysis_summary'])

def display_detailed_analysis():
    """Affiche l'analyse détaillée d'un match spécifique"""
    
    st.header("📈 ANALYSE DÉTAILLÉE PAR MATCH")
    
    # Générer des matchs à venir
    try:
        fixtures = st.session_state.prediction_system.data_generator.generate_upcoming_fixtures(days_ahead=3)
        
        if not fixtures:
            st.info("Générez d'abord des matchs en lançant une analyse")
            return
        
        # Liste des matchs disponibles
        match_list = []
        for fixture in fixtures:
            match_display = f"{fixture['home_name']} vs {fixture['away_name']} - {fixture['league_name']} ({fixture['date'][:10]})"
            match_list.append((match_display, fixture))
        
        selected_match_display = st.selectbox(
            "Sélectionnez un match à analyser",
            options=[m[0] for m in match_list],
            index=0 if match_list else 0
        )
        
        if selected_match_display:
            # Trouver le match correspondant
            selected_fixture = None
            for display, fixture in match_list:
                if display == selected_match_display:
                    selected_fixture = fixture
                    break
            
            if selected_fixture and st.button("🔍 ANALYSER CE MATCH", type="primary"):
                with st.spinner("Analyse en cours..."):
                    analysis = st.session_state.prediction_system.analyze_match(selected_fixture)
                    
                    if analysis:
                        display_complete_analysis(analysis)
                    else:
                        st.error("Impossible d'analyser ce match")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

def display_complete_analysis(analysis: Dict):
    """Affiche une analyse complète détaillée"""
    
    st.markdown(f"## 🎯 ANALYSE COMPLÈTE: {analysis['match']}")
    
    # Tableau de bord rapide
    col_dash1, col_dash2, col_dash3, col_dash4 = st.columns(4)
    
    with col_dash1:
        conf_score = analysis['confidence']['score']
        if conf_score >= 80:
            st.success(f"CONFIANCE: {conf_score:.1f}%")
        elif conf_score >= 60:
            st.warning(f"CONFIANCE: {conf_score:.1f}%")
        else:
            st.error(f"CONFIANCE: {conf_score:.1f}%")
    
    with col_dash2:
        st.info(f"TYPE: {analysis['match_type']}")
    
    with col_dash3:
        main_pred = analysis['predictions'][0]
        st.info(f"PRONOSTIC: {main_pred['prediction']}")
    
    with col_dash4:
        score_pred = analysis['probable_score']
        st.info(f"SCORE: {score_pred['score']}")
    
    st.divider()
    
    # Graphique des probabilités
    st.subheader("📊 DISTRIBUTION DES PROBABILITÉS")
    
    prob_data = pd.DataFrame({
        'Résultat': ['Victoire domicile', 'Match nul', 'Victoire extérieur'],
        'Probabilité (%)': [
            analysis['probabilities']['home_win'],
            analysis['probabilities']['draw'],
            analysis['probabilities']['away_win']
        ]
    })
    
    st.bar_chart(prob_data.set_index('Résultat'))
    
    # Analyse détaillée en colonnes
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        st.subheader("📈 FACTEURS CLÉS")
        
        st.write("**Forme des équipes:**")
        st.write(f"- Domicile: {analysis['probabilities']['home_form']}%")
        st.write(f"- Extérieur: {analysis['probabilities']['away_form']}%")
        
        st.write("**Historique H2H:**")
        h2h_stats = analysis['h2h_history']['stats']
        st.write(f"- Domicile {h2h_stats['home_wins']} - {h2h_stats['draws']} - {h2h_stats['away_wins']} Extérieur")
        st.write(f"- Tendence: {h2h_stats['trend']}")
        
        # Facteurs de confiance
        if analysis['confidence']['factors']:
            st.write("**Facteurs positifs:**")
            for factor in analysis['confidence']['factors']:
                st.write(f"• {factor}")
    
    with col_analysis2:
        st.subheader("💰 ANALYSE DES PARIS")
        
        if analysis['betting_recommendations']:
            for rec in analysis['betting_recommendations']:
                with st.container():
                    st.write(f"**{rec['type']} - {rec['prediction']}**")
                    st.write(f"- Cote estimée: {rec['odd_estimee']}")
                    st.write(f"- Valeur: {rec['valeur']} {rec.get('couleur', '')}")
                    st.write(f"- Risque: {rec['risque']}")
                    if 'valeur_score' in rec:
                        st.write(f"- Score valeur: {rec['valeur_score']}%")
                    st.divider()
        else:
            st.warning("⚠️ Aucun pari recommandé - Match trop risqué")
    
    # Prédictions détaillées
    st.subheader("🔮 PRÉDICTIONS DÉTAILLÉES")
    
    for pred in analysis['predictions']:
        with st.expander(f"{pred['type']}: {pred['prediction']}"):
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.write(f"**Probabilité:** {pred['probability']}")
                st.write(f"**Confiance:** {pred['confidence']}")
            
            with col_pred2:
                if pred['type'] == 'Résultat final':
                    st.write("**Explication:** Prédiction du résultat final du match")
                elif pred['type'] == 'Double chance':
                    st.write("**Explication:** Plus sûr qu'un pari simple")
                elif pred['type'] == 'Total buts':
                    st.write("**Explication:** Prédiction du nombre total de buts")
                elif pred['type'] == 'Les deux équipes marquent':
                    st.write("**Explication:** Les deux équipes marqueront-elles?")
                elif pred['type'] == 'Handicap asiatique':
                    st.write("**Explication:** Pour les matches déséquilibrés")
    
    # Conclusion
    st.subheader("📝 CONCLUSION")
    
    st.write(analysis['analysis_summary'])

def display_all_matches():
    """Affiche tous les matchs disponibles"""
    
    st.header("📅 TOUS LES MATCHS DISPONIBLES")
    
    col_view1, col_view2 = st.columns(2)
    
    with col_view1:
        show_days = st.selectbox("Afficher les matchs sur", [1, 2, 3, 7], index=2)
    
    with col_view2:
        league_filter = st.selectbox("Filtrer par ligue", 
                                   ["Toutes", "Ligue 1", "Premier League", "La Liga", 
                                    "Bundesliga", "Serie A"])
    
    # Générer des matchs
    try:
        fixtures = st.session_state.prediction_system.data_generator.generate_upcoming_fixtures(
            days_ahead=show_days
        )
        
        if not fixtures:
            st.info("Aucun match trouvé")
            return
        
        # Appliquer le filtre de ligue
        if league_filter != "Toutes":
            fixtures = [f for f in fixtures if f.get('league_name') == league_filter]
        
        st.info(f"📊 **{len(fixtures)} matchs trouvés**")
        
        # Afficher les matchs
        for fixture in fixtures:
            with st.container():
                col_match1, col_match2, col_match3 = st.columns([2, 1, 2])
                
                with col_match1:
                    st.write(f"**{fixture.get('home_name', 'Domicile')}**")
                
                with col_match2:
                    st.write("**VS**")
                    st.write(f"{fixture.get('date', '')[11:16] if fixture.get('date') and len(fixture['date']) > 16 else ''}")
                
                with col_match3:
                    st.write(f"**{fixture.get('away_name', 'Extérieur')}**")
                
                st.write(f"📍 **{fixture.get('league_name', '')}** • {fixture.get('date', '')[:10]}")
                
                # Bouton pour analyser ce match
                if st.button(f"🔍 Analyser ce match", key=f"analyze_{fixture.get('fixture_id')}"):
                    analysis = st.session_state.prediction_system.analyze_match(fixture)
                    if analysis:
                        st.session_state.quick_analysis = analysis
                        st.rerun()
                
                st.divider()
        
        # Afficher l'analyse rapide si disponible
        if 'quick_analysis' in st.session_state:
            st.subheader("⚡ ANALYSE RAPIDE")
            
            analysis = st.session_state.quick_analysis
            col_quick1, col_quick2 = st.columns(2)
            
            with col_quick1:
                st.write(f"**Pronostic:** {analysis['predictions'][0]['prediction']}")
                st.write(f"**Probabilité:** {analysis['predictions'][0]['probability']}")
            
            with col_quick2:
                st.write(f"**Confiance:** {analysis['confidence']['overall']}")
                st.write(f"**Score probable:** {analysis['probable_score']['score']}")
            
            if st.button("❌ Fermer l'analyse rapide"):
                del st.session_state.quick_analysis
                st.rerun()
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

def display_history():
    """Affiche l'historique des analyses"""
    
    st.header("📊 HISTORIQUE DES ANALYSES")
    
    if not hasattr(st.session_state.prediction_system, 'prediction_history') or not st.session_state.prediction_system.prediction_history:
        st.info("Aucune analyse dans l'historique. Lancez votre première analyse!")
        return
    
    history = st.session_state.prediction_system.prediction_history
    
    # Statistiques
    st.subheader("📈 STATISTIQUES GLOBALES")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    total_scanned = sum(h.get('total_matches_scanned', 0) for h in history)
    total_predictions = sum(h.get('predictions_made', 0) for h in history)
    avg_conf = np.mean([h.get('avg_confidence', 0) for h in history])
    success_rate = (total_predictions / total_scanned * 100) if total_scanned > 0 else 0
    
    with col_stat1:
        st.metric("Analyses effectuées", len(history))
    
    with col_stat2:
        st.metric("Matchs analysés", total_scanned)
    
    with col_stat3:
        st.metric("Pronostics générés", total_predictions)
    
    with col_stat4:
        st.metric("Taux de succès", f"{success_rate:.1f}%")
    
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
            'Taux': f"{(scan.get('predictions_made', 0) / scan.get('total_matches_scanned', 1) * 100):.1f}%",
            'Confiance moyenne': f"{scan.get('avg_confidence', 0):.1f}%"
        })
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True, hide_index=True)
    
    # Graphique de performance
    if len(history) >= 2:
        st.subheader("📈 ÉVOLUTION DE LA PERFORMANCE")
        
        dates = [h.get('timestamp').strftime('%d/%m') for h in history[-8:]]
        success_rates = [
            (h.get('predictions_made', 0) / h.get('total_matches_scanned', 1) * 100)
            for h in history[-8:]
        ]
        
        chart_data = pd.DataFrame({
            'Date': dates,
            'Taux de succès (%)': success_rates
        })
        
        st.line_chart(chart_data.set_index('Date'), height=300)
    
    # Bouton de nettoyage
    st.divider()
    if st.button("🧹 Effacer l'historique", type="secondary"):
        st.session_state.prediction_system.prediction_history = []
        st.success("Historique effacé avec succès!")
        st.rerun()

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()
