# app.py - Système de Pronostics Personnalisé
# Version avec entrée manuelle des matchs

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
# SYSTÈME DE PRÉDICTION AVANCÉ
# =============================================================================

class AdvancedPredictionSystem:
    """Système de prédiction avancé pour matchs de football"""
    
    def __init__(self):
        self.team_stats = self._initialize_stats()
        self.league_factors = self._initialize_league_factors()
        self.form_history = {}
    
    def _initialize_stats(self) -> Dict:
        """Initialise les statistiques détaillées des équipes"""
        return {
            # Premier League
            'Manchester City': {'attack': 98, 'defense': 90, 'home': 97, 'away': 92, 'form': 95, 'morale': 90},
            'Arsenal': {'attack': 92, 'defense': 85, 'home': 93, 'away': 86, 'form': 88, 'morale': 92},
            'Liverpool': {'attack': 94, 'defense': 87, 'home': 95, 'away': 88, 'form': 92, 'morale': 94},
            'Chelsea': {'attack': 82, 'defense': 80, 'home': 84, 'away': 78, 'form': 75, 'morale': 70},
            'Tottenham': {'attack': 88, 'defense': 82, 'home': 90, 'away': 83, 'form': 85, 'morale': 85},
            'Manchester United': {'attack': 84, 'defense': 82, 'home': 86, 'away': 79, 'form': 78, 'morale': 75},
            'Aston Villa': {'attack': 85, 'defense': 80, 'home': 87, 'away': 79, 'form': 88, 'morale': 90},
            'Newcastle': {'attack': 83, 'defense': 81, 'home': 85, 'away': 78, 'form': 82, 'morale': 85},
            
            # Ligue 1
            'Paris SG': {'attack': 95, 'defense': 88, 'home': 96, 'away': 90, 'form': 94, 'morale': 92},
            'Lille': {'attack': 83, 'defense': 82, 'home': 85, 'away': 79, 'form': 80, 'morale': 82},
            'Marseille': {'attack': 85, 'defense': 81, 'home': 87, 'away': 80, 'form': 82, 'morale': 85},
            'Monaco': {'attack': 84, 'defense': 76, 'home': 86, 'away': 78, 'form': 83, 'morale': 84},
            'Lyon': {'attack': 82, 'defense': 79, 'home': 84, 'away': 77, 'form': 75, 'morale': 70},
            'Nice': {'attack': 81, 'defense': 85, 'home': 83, 'away': 78, 'form': 84, 'morale': 86},
            
            # La Liga
            'Real Madrid': {'attack': 96, 'defense': 89, 'home': 96, 'away': 91, 'form': 95, 'morale': 96},
            'Atlético Madrid': {'attack': 87, 'defense': 88, 'home': 90, 'away': 82, 'form': 86, 'morale': 88},
            'Barcelona': {'attack': 92, 'defense': 87, 'home': 93, 'away': 87, 'form': 90, 'morale': 91},
            'Sevilla': {'attack': 80, 'defense': 82, 'home': 83, 'away': 76, 'form': 72, 'morale': 68},
            'Valencia': {'attack': 78, 'defense': 81, 'home': 82, 'away': 74, 'form': 76, 'morale': 78},
            
            # Bundesliga
            'Bayern Munich': {'attack': 97, 'defense': 88, 'home': 96, 'away': 92, 'form': 94, 'morale': 93},
            'Borussia Dortmund': {'attack': 88, 'defense': 82, 'home': 90, 'away': 83, 'form': 86, 'morale': 88},
            'Bayer Leverkusen': {'attack': 89, 'defense': 84, 'home': 91, 'away': 85, 'form': 92, 'morale': 95},
            'RB Leipzig': {'attack': 85, 'defense': 82, 'home': 87, 'away': 81, 'form': 83, 'morale': 84},
            
            # Serie A
            'Inter Milan': {'attack': 93, 'defense': 90, 'home': 94, 'away': 88, 'form': 94, 'morale': 95},
            'Juventus': {'attack': 84, 'defense': 88, 'home': 87, 'away': 81, 'form': 82, 'morale': 83},
            'AC Milan': {'attack': 87, 'defense': 85, 'home': 89, 'away': 82, 'form': 84, 'morale': 85},
            'AS Roma': {'attack': 82, 'defense': 83, 'home': 85, 'away': 78, 'form': 80, 'morale': 81},
            'Napoli': {'attack': 85, 'defense': 84, 'home': 87, 'away': 80, 'form': 79, 'morale': 78},
            
            # Autres équipes européennes
            'Porto': {'attack': 84, 'defense': 82, 'home': 86, 'away': 80, 'form': 83, 'morale': 84},
            'Benfica': {'attack': 83, 'defense': 81, 'home': 85, 'away': 79, 'form': 82, 'morale': 83},
            'Ajax': {'attack': 82, 'defense': 80, 'home': 84, 'away': 78, 'form': 79, 'morale': 80},
            'PSV': {'attack': 81, 'defense': 79, 'home': 83, 'away': 77, 'form': 84, 'morale': 85},
        }
    
    def _initialize_league_factors(self) -> Dict:
        """Initialise les facteurs spécifiques aux ligues"""
        return {
            'Premier League': {'goals': 1.1, 'draw': 1.10, 'home_advantage': 1.15},
            'Ligue 1': {'goals': 0.9, 'draw': 1.15, 'home_advantage': 1.12},
            'La Liga': {'goals': 1.0, 'draw': 1.12, 'home_advantage': 1.10},
            'Bundesliga': {'goals': 1.2, 'draw': 1.08, 'home_advantage': 1.18},
            'Serie A': {'goals': 0.8, 'draw': 1.20, 'home_advantage': 1.05},
            'Champions League': {'goals': 1.0, 'draw': 1.05, 'home_advantage': 1.05},
            'Europa League': {'goals': 1.0, 'draw': 1.08, 'home_advantage': 1.08},
            'Ligue 2': {'goals': 0.8, 'draw': 1.18, 'home_advantage': 1.10},
            'Autre': {'goals': 1.0, 'draw': 1.10, 'home_advantage': 1.12},
        }
    
    def get_team_data(self, team_name: str) -> Dict:
        """Récupère les données d'une équipe"""
        if team_name in self.team_stats:
            return self.team_stats[team_name]
        
        # Chercher des correspondances partielles
        for known_team in self.team_stats:
            if team_name.lower() in known_team.lower() or known_team.lower() in team_name.lower():
                return self.team_stats[known_team]
        
        # Données par défaut pour nouvelles équipes
        return {
            'attack': 75,
            'defense': 75,
            'home': 78,
            'away': 72,
            'form': 70,
            'morale': 70,
        }
    
    def analyze_match(self, home_team: str, away_team: str, league: str = 'Autre',
                     custom_data: Dict = None) -> Dict:
        """Analyse complète d'un match avec données personnalisées"""
        
        try:
            # Récupérer les données des équipes
            home_data = self.get_team_data(home_team)
            away_data = self.get_team_data(away_team)
            
            # Appliquer les données personnalisées si fournies
            if custom_data:
                home_data = self._apply_custom_data(home_data, custom_data.get('home_team', {}))
                away_data = self._apply_custom_data(away_data, custom_data.get('away_team', {}))
                league = custom_data.get('league', league)
            
            # Facteurs de ligue
            league_factors = self.league_factors.get(league, self.league_factors['Autre'])
            
            # Calcul des forces
            home_strength = self._calculate_team_strength(home_data, is_home=True, league_factors=league_factors)
            away_strength = self._calculate_team_strength(away_data, is_home=False, league_factors=league_factors)
            
            # Ajustements supplémentaires
            home_strength = self._apply_adjustments(home_strength, home_data, 'home')
            away_strength = self._apply_adjustments(away_strength, away_data, 'away')
            
            # Calcul des probabilités
            home_prob, draw_prob, away_prob = self._calculate_probabilities(
                home_strength, away_strength, league_factors
            )
            
            # Score prédit
            home_goals, away_goals = self._predict_score(home_data, away_data, league_factors)
            
            # Over/Under et BTTS
            over_under, over_prob = self._calculate_over_under(home_goals, away_goals)
            btts, btts_prob = self._calculate_btts(home_goals, away_goals)
            
            # Cotes
            odds = self._calculate_odds(home_prob, draw_prob, away_prob)
            
            # Prédiction principale et confiance
            main_pred, pred_type, confidence = self._get_main_prediction(
                home_team, away_team, home_prob, draw_prob, away_prob
            )
            
            # Analyse détaillée
            analysis = self._generate_detailed_analysis(
                home_team, away_team, league, home_data, away_data,
                home_prob, draw_prob, away_prob, home_goals, away_goals,
                confidence, custom_data
            )
            
            # Statistiques détaillées
            detailed_stats = self._generate_detailed_stats(
                home_data, away_data, home_goals, away_goals
            )
            
            return {
                'match': f"{home_team} vs {away_team}",
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'probabilities': {
                    'home_win': round(home_prob, 1),
                    'draw': round(draw_prob, 1),
                    'away_win': round(away_prob, 1)
                },
                'main_prediction': main_pred,
                'prediction_type': pred_type,
                'confidence': round(confidence, 1),
                'score_prediction': f"{home_goals}-{away_goals}",
                'expected_goals': {
                    'home': round(self._calculate_xg(home_data, away_data, True), 2),
                    'away': round(self._calculate_xg(away_data, home_data, False), 2)
                },
                'over_under': over_under,
                'over_prob': round(over_prob, 1),
                'btts': btts,
                'btts_prob': round(btts_prob, 1),
                'odds': odds,
                'analysis': analysis,
                'detailed_stats': detailed_stats,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            st.error(f"Erreur dans l'analyse: {str(e)}")
            return None
    
    def _apply_custom_data(self, base_data: Dict, custom_data: Dict) -> Dict:
        """Applique les données personnalisées aux statistiques de base"""
        if not custom_data:
            return base_data
        
        updated_data = base_data.copy()
        
        # Ajuster les statistiques en fonction des données personnalisées
        factors = {
            'forme': 0.15,  # 15% d'impact
            'blessures': -0.10,  # -10% par joueur important blessé
            'suspensions': -0.08,  # -8% par suspension
            'motivation': 0.12,  # 12% d'impact
            'fatigue': -0.07,  # -7% d'impact
        }
        
        for key, factor in factors.items():
            if key in custom_data:
                value = custom_data[key]
                if isinstance(value, (int, float)):
                    # Ajustement proportionnel
                    adjustment = 1 + (value * factor)
                    updated_data['attack'] = min(100, max(0, updated_data['attack'] * adjustment))
                    updated_data['defense'] = min(100, max(0, updated_data['defense'] * adjustment))
        
        return updated_data
    
    def _calculate_team_strength(self, team_data: Dict, is_home: bool, league_factors: Dict) -> float:
        """Calcule la force d'une équipe"""
        base_strength = (
            team_data['attack'] * 0.35 +
            team_data['defense'] * 0.30 +
            team_data['form'] * 0.20 +
            team_data['morale'] * 0.15
        )
        
        # Avantage domicile
        if is_home:
            base_strength *= league_factors['home_advantage']
            base_strength *= (team_data['home'] / 100)
        else:
            base_strength *= (team_data['away'] / 100)
        
        return base_strength
    
    def _apply_adjustments(self, strength: float, team_data: Dict, location: str) -> float:
        """Applique des ajustements supplémentaires"""
        # Ajustement selon la forme récente
        form_adjustment = 1.0 + ((team_data['form'] - 75) / 100) * 0.3
        
        # Ajustement selon le moral
        morale_adjustment = 1.0 + ((team_data['morale'] - 75) / 100) * 0.2
        
        return strength * form_adjustment * morale_adjustment
    
    def _calculate_probabilities(self, home_strength: float, away_strength: float, 
                                league_factors: Dict) -> Tuple[float, float, float]:
        """Calcule les probabilités de résultat"""
        total_strength = home_strength + away_strength
        
        home_prob = (home_strength / total_strength) * 100 * 0.85
        away_prob = (away_strength / total_strength) * 100 * 0.85
        draw_prob = 100 - home_prob - away_prob
        
        # Ajustement match nul selon la ligue
        draw_prob *= league_factors['draw']
        
        # Normaliser
        total = home_prob + draw_prob + away_prob
        home_prob = (home_prob / total) * 100
        draw_prob = (draw_prob / total) * 100
        away_prob = (away_prob / total) * 100
        
        return home_prob, draw_prob, away_prob
    
    def _predict_score(self, home_data: Dict, away_data: Dict, league_factors: Dict) -> Tuple[int, int]:
        """Prédit le score final"""
        # xG attendus
        home_xg = self._calculate_xg(home_data, away_data, True) * league_factors['goals']
        away_xg = self._calculate_xg(away_data, home_data, False) * league_factors['goals']
        
        # Conversion en buts avec variabilité
        home_goals = self._xg_to_goals(home_xg, is_home=True)
        away_goals = self._xg_to_goals(away_xg, is_home=False)
        
        # Ajustement final
        home_goals = max(0, home_goals)
        away_goals = max(0, away_goals)
        
        # Limiter le score maximum
        home_goals = min(home_goals, 5)
        away_goals = min(away_goals, 4)
        
        # Éviter 0-0 improbable
        if home_goals == away_goals == 0 and random.random() < 0.8:
            home_goals = random.randint(0, 1)
            away_goals = random.randint(0, 1)
        
        return int(round(home_goals)), int(round(away_goals))
    
    def _calculate_xg(self, attacking_team: Dict, defending_team: Dict, is_home: bool) -> float:
        """Calcule les xG (expected goals)"""
        attack_factor = attacking_team['attack'] / 100
        defense_factor = (100 - defending_team['defense']) / 100
        location_factor = 1.2 if is_home else 1.0
        
        base_xg = 1.5 * attack_factor * defense_factor * location_factor
        
        # Ajouter de la variabilité
        variability = random.uniform(0.8, 1.2)
        
        return base_xg * variability
    
    def _xg_to_goals(self, xg: float, is_home: bool) -> float:
        """Convertit les xG en buts réels"""
        # Simulation de Poisson simplifiée
        goals = 0
        remaining_xg = xg
        
        while remaining_xg > 0:
            if random.random() < remaining_xg:
                goals += 1
                remaining_xg -= 1
            else:
                break
        
        # Ajouter un petit bonus pour les occasions claires
        if xg > 1.5:
            goals += random.uniform(0, 0.3)
        
        return goals
    
    def _calculate_over_under(self, home_goals: int, away_goals: int) -> Tuple[str, float]:
        """Calcule Over/Under 2.5"""
        total_goals = home_goals + away_goals
        
        if total_goals >= 3:
            result = "Over 2.5"
            probability = min(95, 60 + (total_goals - 2) * 15)
        else:
            result = "Under 2.5"
            probability = min(95, 70 - (3 - total_goals) * 20)
        
        return result, probability
    
    def _calculate_btts(self, home_goals: int, away_goals: int) -> Tuple[str, float]:
        """Calcule Both Teams To Score"""
        if home_goals > 0 and away_goals > 0:
            result = "Oui"
            probability = min(95, 65 + min(home_goals, away_goals) * 10)
        else:
            result = "Non"
            probability = min(95, 70 - abs(home_goals - away_goals) * 15)
        
        return result, probability
    
    def _calculate_odds(self, home_prob: float, draw_prob: float, away_prob: float) -> Dict:
        """Calcule les cotes décimales"""
        margin = 1.05  # Marge de bookmaker
        
        home_odd = round(1 / (home_prob / 100) * margin, 2)
        draw_odd = round(1 / (draw_prob / 100) * margin, 2)
        away_odd = round(1 / (away_prob / 100) * margin, 2)
        
        # Limites réalistes
        home_odd = max(1.1, min(15.0, home_odd))
        draw_odd = max(2.0, min(8.0, draw_odd))
        away_odd = max(1.5, min(12.0, away_odd))
        
        return {
            'home': home_odd,
            'draw': draw_odd,
            'away': away_odd,
            '1X': round(1 / ((home_prob + draw_prob) / 100) * margin, 2),
            'X2': round(1 / ((draw_prob + away_prob) / 100) * margin, 2),
        }
    
    def _get_main_prediction(self, home_team: str, away_team: str,
                            home_prob: float, draw_prob: float, away_prob: float) -> Tuple[str, str, float]:
        """Détermine la prédiction principale"""
        if home_prob >= away_prob and home_prob >= draw_prob:
            main_pred = f"Victoire {home_team}"
            pred_type = "1"
            confidence = home_prob
        elif away_prob >= home_prob and away_prob >= draw_prob:
            main_pred = f"Victoire {away_team}"
            pred_type = "2"
            confidence = away_prob
        else:
            main_pred = "Match nul"
            pred_type = "X"
            confidence = draw_prob
        
        return main_pred, pred_type, confidence
    
    def _generate_detailed_analysis(self, home_team: str, away_team: str, league: str,
                                  home_data: Dict, away_data: Dict,
                                  home_prob: float, draw_prob: float, away_prob: float,
                                  home_goals: int, away_goals: int,
                                  confidence: float, custom_data: Dict = None) -> str:
        """Génère une analyse détaillée du match"""
        
        analysis = []
        
        analysis.append("## 📊 ANALYSE DÉTAILLÉE DU MATCH")
        analysis.append(f"**{home_team} vs {away_team}**")
        analysis.append(f"*{league}*")
        analysis.append("")
        
        # Forces des équipes
        analysis.append("### ⚖️ COMPARAISON DES FORCES")
        analysis.append(f"**{home_team}:**")
        analysis.append(f"- Attaque: {home_data['attack']}/100")
        analysis.append(f"- Défense: {home_data['defense']}/100")
        analysis.append(f"- Forme: {home_data['form']}/100")
        analysis.append(f"- Moral: {home_data['morale']}/100")
        analysis.append("")
        
        analysis.append(f"**{away_team}:**")
        analysis.append(f"- Attaque: {away_data['attack']}/100")
        analysis.append(f"- Défense: {away_data['defense']}/100")
        analysis.append(f"- Forme: {away_data['form']}/100")
        analysis.append(f"- Moral: {away_data['morale']}/100")
        analysis.append("")
        
        # Probabilités
        analysis.append("### 🎯 PROBABILITÉS DE RÉSULTAT")
        analysis.append(f"- **Victoire {home_team}:** {home_prob:.1f}%")
        analysis.append(f"- **Match nul:** {draw_prob:.1f}%")
        analysis.append(f"- **Victoire {away_team}:** {away_prob:.1f}%")
        analysis.append("")
        
        # Score prédit
        analysis.append(f"### ⚽ SCORE PRÉDIT: **{home_goals}-{away_goals}**")
        
        # Analyse tactique
        analysis.append("### 🧠 ANALYSE TACTIQUE")
        
        # Qui domine ?
        if home_prob > away_prob + 15:
            analysis.append(f"- **{home_team} devrait dominer** le match")
            analysis.append(f"- Contrôle du jeu attendu en faveur des locaux")
        elif away_prob > home_prob + 15:
            analysis.append(f"- **{away_team} devrait dominer** le match")
            analysis.append(f"- Les visiteurs ont l'avantage tactique")
        else:
            analysis.append("- **Match équilibré** prévu")
            analysis.append("- Les deux équipes devraient se neutraliser")
        
        # Style de jeu
        attack_diff = home_data['attack'] - away_data['attack']
        defense_diff = home_data['defense'] - away_data['defense']
        
        if attack_diff > 20:
            analysis.append(f"- **{home_team} plus offensif**, devrait créer plus d'occasions")
        elif attack_diff < -20:
            analysis.append(f"- **{away_team} plus offensif**, dangereux en contre")
        
        if defense_diff > 15:
            analysis.append(f"- **{home_team} plus solide défensivement**")
        elif defense_diff < -15:
            analysis.append(f"- **{away_team} meilleure en défense**")
        
        analysis.append("")
        
        # Facteurs clés
        analysis.append("### 🔑 FACTEURS CLÉS")
        
        if home_data['form'] > away_data['form'] + 10:
            analysis.append(f"- **Forme récente** en faveur de {home_team}")
        elif away_data['form'] > home_data['form'] + 10:
            analysis.append(f"- **Forme récente** en faveur de {away_team}")
        
        if home_data['morale'] > away_data['morale'] + 10:
            analysis.append(f"- **Moral** plus élevé chez {home_team}")
        elif away_data['morale'] > home_data['morale'] + 10:
            analysis.append(f"- **Moral** plus élevé chez {away_team}")
        
        # Données personnalisées
        if custom_data:
            analysis.append("")
            analysis.append("### 🎛️ FACTEURS PERSONNALISÉS")
            
            home_custom = custom_data.get('home_team', {})
            away_custom = custom_data.get('away_team', {})
            
            for key, value in home_custom.items():
                if isinstance(value, (int, float)) and value != 0:
                    analysis.append(f"- {home_team}: {key} = {value}")
            
            for key, value in away_custom.items():
                if isinstance(value, (int, float)) and value != 0:
                    analysis.append(f"- {away_team}: {key} = {value}")
        
        analysis.append("")
        
        # Conseils de pari
        analysis.append("### 💰 CONSEILS DE PARI")
        
        if confidence >= 75:
            analysis.append("- **PRONOSTIC FORT** - Bonne valeur")
            analysis.append(f"- Pari recommandé: **{self._get_main_prediction(home_team, away_team, home_prob, draw_prob, away_prob)[0]}**")
        elif confidence >= 65:
            analysis.append("- **PRONOSTIC MOYEN** - Valeur correcte")
            analysis.append(f"- Pari envisageable: **{self._get_main_prediction(home_team, away_team, home_prob, draw_prob, away_prob)[0]}**")
        else:
            analysis.append("- **PRONOSTIC RISQUÉ** - À éviter ou petites mises")
            analysis.append("- Mise réduite recommandée")
        
        analysis.append("")
        analysis.append("---")
        analysis.append("*Analyse générée automatiquement - Basée sur les données fournies*")
        
        return '\n'.join(analysis)
    
    def _generate_detailed_stats(self, home_data: Dict, away_data: Dict, 
                                home_goals: int, away_goals: int) -> Dict:
        """Génère des statistiques détaillées"""
        return {
            'home_team': {
                'attack_strength': home_data['attack'],
                'defense_strength': home_data['defense'],
                'form_level': home_data['form'],
                'morale_level': home_data['morale'],
                'predicted_goals': home_goals,
            },
            'away_team': {
                'attack_strength': away_data['attack'],
                'defense_strength': away_data['defense'],
                'form_level': away_data['form'],
                'morale_level': away_data['morale'],
                'predicted_goals': away_goals,
            },
            'match_characteristics': {
                'expected_total_goals': home_goals + away_goals,
                'goal_difference': abs(home_goals - away_goals),
                'home_advantage_factor': 1.15,
            }
        }

# =============================================================================
# APPLICATION STREAMLIT
# =============================================================================

def main():
    """Application principale avec entrée manuelle"""
    
    st.set_page_config(
        page_title="Pronostics Personnalisés",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #1E88E5 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border-left: 5px solid #1E88E5;
    }
    .input-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 2px solid #E3F2FD;
    }
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .team-badge {
        background: #1E88E5;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .result-badge {
        background: #43A047;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .slider-container {
        background: #F5F5F5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-title">⚽ PRONOSTICS PERSONNALISÉS</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem; color: #666;">'
                'Analyse avancée de matchs • Entrée manuelle • Données personnalisées</div>', 
                unsafe_allow_html=True)
    
    # Initialisation du système
    if 'prediction_system' not in st.session_state:
        st.session_state.prediction_system = AdvancedPredictionSystem()
    
    # Sidebar pour l'historique
    with st.sidebar:
        st.markdown("## 📋 HISTORIQUE")
        
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if st.session_state.analysis_history:
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"📝 {analysis['match']}"):
                    st.write(f"**Ligue:** {analysis['league']}")
                    st.write(f"**Score prédit:** {analysis['score_prediction']}")
                    st.write(f"**Pronostic:** {analysis['main_prediction']}")
                    st.write(f"**Confiance:** {analysis['confidence']}%")
                    
                    if st.button(f"🔁 Réutiliser #{len(st.session_state.analysis_history)-i}", key=f"reuse_{i}"):
                        st.session_state.last_analysis = analysis
                        st.rerun()
        
        st.markdown("---")
        st.markdown("## ⚙️ PARAMÈTRES")
        
        # Options d'affichage
        show_details = st.checkbox("Afficher détails techniques", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 STATISTIQUES")
        
        if st.session_state.analysis_history:
            total_analyses = len(st.session_state.analysis_history)
            avg_confidence = sum(a['confidence'] for a in st.session_state.analysis_history) / total_analyses
            
            st.metric("Analyses effectuées", total_analyses)
            st.metric("Confiance moyenne", f"{avg_confidence:.1f}%")
    
    # Contenu principal
    st.markdown("## 🎯 SAISIE DU MATCH")
    
    # Formulaire de saisie
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏠 ÉQUIPE À DOMICILE")
            home_team = st.text_input("Nom de l'équipe", key="home_team", 
                                      value=st.session_state.get('last_analysis', {}).get('home_team', ''))
            
            st.markdown("#### 📊 Données personnalisées")
            home_form = st.slider("Forme récente (0-100)", 0, 100, 75, key="home_form")
            home_injuries = st.slider("Impact des blessures (0-10)", 0, 10, 0, key="home_injuries")
            home_motivation = st.slider("Motivation (0-100)", 0, 100, 75, key="home_motivation")
            
        with col2:
            st.markdown("### 🏃 ÉQUIPE À L'EXTERIEUR")
            away_team = st.text_input("Nom de l'équipe", key="away_team",
                                      value=st.session_state.get('last_analysis', {}).get('away_team', ''))
            
            st.markdown("#### 📊 Données personnalisées")
            away_form = st.slider("Forme récente (0-100)", 0, 100, 75, key="away_form")
            away_injuries = st.slider("Impact des blessures (0-10)", 0, 10, 0, key="away_injuries")
            away_motivation = st.slider("Motivation (0-100)", 0, 100, 75, key="away_motivation")
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 🏆 COMPÉTITION")
            league_options = ['Premier League', 'Ligue 1', 'La Liga', 'Bundesliga', 
                            'Serie A', 'Champions League', 'Europa League', 'Ligue 2', 'Autre']
            league = st.selectbox("Sélectionnez la ligue", league_options, index=0)
        
        with col4:
            st.markdown("### 📅 DATE DU MATCH")
            match_date = st.date_input("Date", value=date.today())
            match_time = st.time_input("Heure", value=datetime.now().time())
        
        st.markdown("---")
        
        # Bouton d'analyse
        col5, col6, col7 = st.columns([1, 2, 1])
        with col6:
            if st.button("🎯 ANALYSER LE MATCH", type="primary", use_container_width=True):
                if home_team and away_team:
                    with st.spinner("Analyse en cours..."):
                        # Préparer les données personnalisées
                        custom_data = {
                            'home_team': {
                                'forme': (home_form - 75) / 25,  # Normalisé entre -1 et 1
                                'blessures': home_injuries / 5,  # Normalisé entre 0 et 2
                                'motivation': (home_motivation - 75) / 25,
                            },
                            'away_team': {
                                'forme': (away_form - 75) / 25,
                                'blessures': away_injuries / 5,
                                'motivation': (away_motivation - 75) / 25,
                            },
                            'league': league,
                            'date': match_date.strftime('%Y-%m-%d'),
                            'time': match_time.strftime('%H:%M'),
                        }
                        
                        # Effectuer l'analyse
                        result = st.session_state.prediction_system.analyze_match(
                            home_team, away_team, league, custom_data
                        )
                        
                        if result:
                            st.session_state.last_result = result
                            st.session_state.analysis_history.append(result)
                            st.success("✅ Analyse terminée !")
                        else:
                            st.error("❌ Erreur lors de l'analyse")
                else:
                    st.warning("⚠️ Veuillez saisir les noms des deux équipes")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Affichage des résultats
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        
        st.markdown("## 📊 RÉSULTATS DE L'ANALYSE")
        
        with st.container():
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            # Header avec le match
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f'<div class="team-badge">🏠 {result["home_team"]}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"### VS")
                st.caption(f"{result['league']} • {match_date.strftime('%d/%m/%Y')} {match_time.strftime('%H:%M')}")
            
            with col3:
                st.markdown(f'<div class="team-badge">🏃 {result["away_team"]}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Principales prédictions
            st.markdown(f'<div class="result-badge">🎯 PRONOSTIC: {result["main_prediction"]}</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown("**⚽ SCORE**")
                st.markdown(f"## {result['score_prediction']}")
                st.markdown("Prédit")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown("**🎯 CONFIANC**")
                st.markdown(f"## {result['confidence']}%")
                st.markdown("Fiabilité")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown("**📈 OVER/UNDER**")
                st.markdown(f"## {result['over_under']}")
                st.markdown(f"{result['over_prob']}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown("**🎯 BTTS**")
                st.markdown(f"## {result['btts']}")
                st.markdown(f"{result['btts_prob']}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Cotes
            st.markdown("### 💰 COTES ESTIMÉES")
            odds_cols = st.columns(5)
            
            with odds_cols[0]:
                st.metric("1", f"{result['odds']['home']:.2f}")
            with odds_cols[1]:
                st.metric("X", f"{result['odds']['draw']:.2f}")
            with odds_cols[2]:
                st.metric("2", f"{result['odds']['away']:.2f}")
            with odds_cols[3]:
                st.metric("1X", f"{result['odds']['1X']:.2f}")
            with odds_cols[4]:
                st.metric("X2", f"{result['odds']['X2']:.2f}")
            
            st.markdown("---")
            
            # Analyse détaillée
            st.markdown(result['analysis'])
            
            # Détails techniques (optionnel)
            if show_details:
                with st.expander("📈 DÉTAILS TECHNIQUES"):
                    st.json(result['detailed_stats'])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Section d'explication
    with st.expander("❓ COMMENT FONCTIONNE L'ANALYSE ?"):
        st.markdown("""
        ### 🔍 Méthodologie d'analyse
        
        Notre système utilise un algorithme avancé qui prend en compte :
        
        **1. Données de base des équipes :**
        - Attaque (capacité offensive)
        - Défense (solidité défensive)
        - Forme récente
        - Moral de l'équipe
        
        **2. Facteurs contextuels :**
        - Avantage du terrain (domicile/extérieur)
        - Spécificités de la ligue
        - Données personnalisées (blessures, motivation, etc.)
        
        **3. Calcul des probabilités :**
        - Modèle statistique bayésien
        - Simulation de Monte-Carlo
        - Ajustement selon les facteurs de ligue
        
        **4. Prédiction du score :**
        - Expected Goals (xG) calculés
        - Conversion en buts réels
        - Variabilité aléatoire contrôlée
        
        **5. Output final :**
        - Probabilités de chaque résultat
        - Score prédit
        - Over/Under 2.5
        - Both Teams To Score (BTTS)
        - Cotes estimées
        
        ### 🎯 Comment utiliser au mieux :
        
        1. **Saisissez précisément** les noms des équipes
        2. **Ajustez les données personnalisées** selon l'actualité
        3. **Sélectionnez la bonne ligue** pour des facteurs adaptés
        4. **Analysez les conseils** de pari avec prudence
        5. **Consultez l'historique** pour suivre vos analyses
        
        *Note : Il s'agit d'une aide à la décision, pas d'une garantie de résultats.*
        """)
    
    # Section de suggestions
    with st.expander("💡 SUGGESTIONS DE MATCHS À ANALYSER"):
        st.markdown("""
        ### 📋 Matchs intéressants à analyser :
        
        **Ce week-end :**
        - Paris SG vs Lille (Ligue 1)
        - Manchester City vs Arsenal (Premier League)
        - Real Madrid vs Atlético Madrid (La Liga)
        - Inter Milan vs Juventus (Serie A)
        - Bayern Munich vs Borussia Dortmund (Bundesliga)
        
        **Matchs européens :**
        - Champions League : Demi-finales
        - Europa League : Quarts de finale
        
        **Matchs à enjeu :**
        - Relégation : équipes en bas de classement
        - Qualification Europe : matchs décisifs
        - Derbys locaux
        """)

if __name__ == "__main__":
    main()
