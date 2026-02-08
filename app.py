def display_match_analysis_manual():
    """Analyse manuelle d'un match avec prédictions détaillées"""
    st.header("🔍 ANALYSE DE MATCH MANUELLE")
    
    st.markdown("""
    <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h4 style="color: #1E88E5;">📊 Analyse Complète de Match</h4>
    <p>Entrez les détails d'un match pour obtenir une analyse détaillée avec :</p>
    <ul>
        <li>✅ Prédictions Élo avancées</li>
        <li>✅ Analyses statistiques</li>
        <li>✅ Recommandations de paris</li>
        <li>✅ Calculs de value bets</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulaire de saisie
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Équipe Domicile")
        home_team = st.text_input("Nom de l'équipe domicile", "Manchester City")
        home_league = st.text_input("Ligue (domicile)", "Premier League")
        home_form = st.slider("Forme récente domicile (1-10)", 1, 10, 7)
        home_goals_scored = st.number_input("Buts marqués/moy (dom)", 0.0, 5.0, 2.3, 0.1)
        home_goals_conceded = st.number_input("Buts encaissés/moy (dom)", 0.0, 5.0, 0.8, 0.1)
    
    with col2:
        st.subheader("⚽ Équipe Extérieur")
        away_team = st.text_input("Nom de l'équipe extérieur", "Liverpool")
        away_league = st.text_input("Ligue (extérieur)", "Premier League")
        away_form = st.slider("Forme récente extérieur (1-10)", 1, 10, 6)
        away_goals_scored = st.number_input("Buts marqués/moy (ext)", 0.0, 5.0, 1.9, 0.1)
        away_goals_conceded = st.number_input("Buts encaissés/moy (ext)", 0.0, 5.0, 1.2, 0.1)
    
    # Paramètres du match
    st.subheader("🎯 Paramètres du Match")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        is_neutral = st.checkbox("Terrain neutre", False)
        importance = st.selectbox("Importance du match", 
                                ["Normal", "Coupe", "Dernière journée", "Finale"])
    
    with col4:
        weather = st.selectbox("Conditions météo", 
                             ["Bonnes", "Pluie", "Vent", "Froid", "Chaud"])
        referee_factor = st.slider("Facteur arbitre (1-10)", 1, 10, 5)
    
    with col5:
        home_missing_players = st.number_input("Joueurs absents (dom)", 0, 10, 1)
        away_missing_players = st.number_input("Joueurs absents (ext)", 0, 10, 2)
    
    # Bouton d'analyse
    if st.button("🚀 ANALYSER LE MATCH", type="primary", use_container_width=True):
        with st.spinner("🔍 Analyse en cours..."):
            
            # Simuler des IDs d'équipe (dans une vraie app, vous les récupéreriez de l'API)
            home_id = hash(home_team) % 1000
            away_id = hash(away_team) % 1000
            
            # 1. CALCUL DES ÉLO DYNAMIQUES
            st.subheader("📈 RATINGS ÉLO DYNAMIQUES")
            
            # Base Élo
            home_base_elo = 1850 if "City" in home_team else 1750
            away_base_elo = 1800 if "Liverpool" in away_team else 1700
            
            # Ajustements
            form_adjustment = (home_form - away_form) * 15
            home_adv = 0 if is_neutral else 70
            importance_mult = 1.2 if importance in ["Finale", "Dernière journée"] else 1.0
            
            # Élo final
            home_final_elo = (home_base_elo + form_adjustment + home_adv) * importance_mult
            away_final_elo = away_base_elo * importance_mult
            
            col6, col7 = st.columns(2)
            
            with col6:
                st.metric("🏠 Élo Domicile", f"{home_final_elo:.0f}", 
                         f"Base: {home_base_elo:.0f} + Forme: {form_adjustment:+.0f}")
                
                # Diagramme radar pour l'équipe domicile
                fig_home = go.Figure(data=go.Scatterpolar(
                    r=[home_form, home_goals_scored*10, (5-home_goals_conceded)*10, 
                       (10-home_missing_players)*10, referee_factor*10],
                    theta=['Forme', 'Attaque', 'Défense', 'Effectif', 'Arbitrage'],
                    fill='toself',
                    name=home_team
                ))
                
                fig_home.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title=f"Profil - {home_team}"
                )
                
                st.plotly_chart(fig_home, use_container_width=True)
            
            with col7:
                st.metric("⚽ Élo Extérieur", f"{away_final_elo:.0f}", 
                         f"Base: {away_base_elo:.0f}")
                
                # Diagramme radar pour l'équipe extérieur
                fig_away = go.Figure(data=go.Scatterpolar(
                    r=[away_form, away_goals_scored*10, (5-away_goals_conceded)*10, 
                       (10-away_missing_players)*10, referee_factor*10],
                    theta=['Forme', 'Attaque', 'Défense', 'Effectif', 'Arbitrage'],
                    fill='toself',
                    name=away_team
                ))
                
                fig_away.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    title=f"Profil - {away_team}"
                )
                
                st.plotly_chart(fig_away, use_container_width=True)
            
            # 2. PRÉDICTIONS DÉTAILLÉES
            st.subheader("🎯 PRÉDICTIONS DU MATCH")
            
            # Calcul des probabilités avec modèle personnalisé
            elo_diff = home_final_elo - away_final_elo
            
            # Probabilité victoire domicile (basée sur Élo)
            home_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
            
            # Probabilité match nul (basée sur différence)
            draw_prob = 0.25 * np.exp(-abs(elo_diff) / 200)
            
            # Ajustements supplémentaires
            weather_factor = 0.95 if weather != "Bonnes" else 1.0
            missing_factor = 1.0 - (away_missing_players - home_missing_players) * 0.02
            
            # Probabilités finales
            home_win_adj = home_win_prob * weather_factor * missing_factor
            draw_adj = draw_prob * weather_factor
            away_win_adj = 1 - home_win_adj - draw_adj
            
            # Normaliser
            total = home_win_adj + draw_adj + away_win_adj
            home_win_final = home_win_adj / total
            draw_final = draw_adj / total
            away_win_final = away_win_adj / total
            
            # Affichage des probabilités
            col8, col9, col10 = st.columns(3)
            
            with col8:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>🏠 VICTOIRE<br>{home_team}</h3>
                <h1 style="font-size: 3rem;">{home_win_final*100:.1f}%</h1>
                <p>Cote équivalente: {1/home_win_final:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col9:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>🤝 MATCH NUL</h3>
                <h1 style="font-size: 3rem;">{draw_final*100:.1f}%</h1>
                <p>Cote équivalente: {1/draw_final:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col10:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h3>⚽ VICTOIRE<br>{away_team}</h3>
                <h1 style="font-size: 3rem;">{away_win_final*100:.1f}%</h1>
                <p>Cote équivalente: {1/away_win_final:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 3. PRÉDICTION SCORE
            st.subheader("📊 PRÉDICTION DE SCORE")
            
            # Calcul des buts attendus
            expected_home_goals = (home_goals_scored * 0.7 + away_goals_conceded * 0.3) * (home_form / 7)
            expected_away_goals = (away_goals_scored * 0.6 + home_goals_conceded * 0.4) * (away_form / 7)
            
            # Ajustements
            if weather != "Bonnes":
                expected_home_goals *= 0.9
                expected_away_goals *= 0.9
            
            col11, col12 = st.columns(2)
            
            with col11:
                st.write("**Buts attendus (Poisson):**")
                
                # Distribution Poisson
                home_goal_probs = {}
                away_goal_probs = {}
                
                for goals in range(0, 6):
                    home_prob = stats.poisson.pmf(goals, expected_home_goals)
                    away_prob = stats.poisson.pmf(goals, expected_away_goals)
                    home_goal_probs[goals] = home_prob
                    away_goal_probs[goals] = away_prob
                
                # Affichage distribution
                fig_goals = go.Figure()
                fig_goals.add_trace(go.Bar(
                    x=list(home_goal_probs.keys()),
                    y=list(home_goal_probs.values()),
                    name=f"{home_team}",
                    marker_color='#667eea'
                ))
                fig_goals.add_trace(go.Bar(
                    x=list(away_goal_probs.keys()),
                    y=list(away_goal_probs.values()),
                    name=f"{away_team}",
                    marker_color='#4facfe'
                ))
                
                fig_goals.update_layout(
                    title=f"Distribution des buts attendus",
                    xaxis_title="Nombre de buts",
                    yaxis_title="Probabilité",
                    barmode='group'
                )
                
                st.plotly_chart(fig_goals, use_container_width=True)
            
            with col12:
                # Scores les plus probables
                st.write("**Scores les plus probables:**")
                
                score_probs = []
                for h in range(0, 4):
                    for a in range(0, 4):
                        prob = home_goal_probs[h] * away_goal_probs[a]
                        score_probs.append({
                            'score': f"{h}-{a}",
                            'probabilité': prob,
                            'home': h,
                            'away': a
                        })
                
                score_probs.sort(key=lambda x: x['probabilité'], reverse=True)
                
                for i, score in enumerate(score_probs[:5]):
                    prob_percent = score['probabilité'] * 100
                    st.metric(f"{score['score']}", f"{prob_percent:.2f}%")
            
            # 4. RECOMMANDATIONS DE PARIS
            st.subheader("💰 RECOMMANDATIONS DE PARIS")
            
            # Simulation de cotes du marché
            market_odds = {
                '1': 1.85,  # Cote pour victoire domicile
                'X': 3.50,  # Cote pour match nul
                '2': 4.20,  # Cote pour victoire extérieur
                'BTTS_yes': 1.65,  # Both teams to score - Oui
                'BTTS_no': 2.20,   # Both teams to score - Non
                'Over_2.5': 1.95,  # Over 2.5 buts
                'Under_2.5': 1.85  # Under 2.5 buts
            }
            
            # Calcul des value bets
            st.write("**🔍 Value Bets détectés:**")
            
            value_bets = []
            
            # Analyse 1X2
            if (home_win_final * market_odds['1']) > 1:
                edge = (home_win_final * market_odds['1']) - 1
                value_bets.append({
                    'marché': '1X2',
                    'sélection': f"{home_team} (1)",
                    'cote': market_odds['1'],
                    'probabilité': home_win_final,
                    'edge': f"{edge*100:.2f}%",
                    'recommandation': 'FAIBLE' if edge < 0.05 else 'MOYENNE' if edge < 0.10 else 'FORTE'
                })
            
            if (draw_final * market_odds['X']) > 1:
                edge = (draw_final * market_odds['X']) - 1
                value_bets.append({
                    'marché': '1X2',
                    'sélection': 'Match Nul (X)',
                    'cote': market_odds['X'],
                    'probabilité': draw_final,
                    'edge': f"{edge*100:.2f}%",
                    'recommandation': 'FAIBLE' if edge < 0.05 else 'MOYENNE' if edge < 0.10 else 'FORTE'
                })
            
            # Analyse Both Teams To Score
            prob_btts = 1 - (home_goal_probs[0] * away_goal_probs[0])
            if (prob_btts * market_odds['BTTS_yes']) > 1.05:
                edge = (prob_btts * market_odds['BTTS_yes']) - 1
                value_bets.append({
                    'marché': 'BTTS',
                    'sélection': 'Les deux équipes marquent',
                    'cote': market_odds['BTTS_yes'],
                    'probabilité': prob_btts,
                    'edge': f"{edge*100:.2f}%",
                    'recommandation': 'FAIBLE' if edge < 0.05 else 'MOYENNE' if edge < 0.10 else 'FORTE'
                })
            
            # Afficher les value bets
            if value_bets:
                df_value = pd.DataFrame(value_bets)
                st.dataframe(df_value, use_container_width=True, hide_index=True)
                
                # Recommandation principale
                best_bet = max(value_bets, key=lambda x: float(x['edge'].replace('%', '')))
                
                st.success(f"""
                🎯 **MEILLEURE OPPORTUNITÉ:** {best_bet['sélection']}
                • **Marché:** {best_bet['marché']}
                • **Cote:** {best_bet['cote']:.2f}
                • **Edge:** {best_bet['edge']}
                • **Recommandation:** {best_bet['recommandation']}
                """)
            else:
                st.warning("⚠️ Aucun value bet significatif détecté avec les cotes actuelles")
            
            # 5. RÉSUMÉ ET CONSEILS
            st.subheader("📋 RÉSUMÉ DE L'ANALYSE")
            
            col13, col14 = st.columns(2)
            
            with col13:
                st.markdown(f"""
                <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <h4>✅ POINTS FORTS {home_team}:</h4>
                <ul>
                    <li>Forme récente: {home_form}/10</li>
                    <li>Attaque domicile: {home_goals_scored} buts/moy</li>
                    <li>Défense solide: {home_goals_conceded} buts/moy</li>
                    <li>Avantage terrain: {70 if not is_neutral else 0} points Élo</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col14:
                st.markdown(f"""
                <div style="background: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #FF9800;">
                <h4>⚠️ RISQUES IDENTIFIÉS:</h4>
                <ul>
                    <li>Joueurs absents: {home_missing_players} (dom) / {away_missing_players} (ext)</li>
                    <li>Météo: {weather}</li>
                    <li>Importance: {importance}</li>
                    <li>Confiance modèle: {min(100, int(abs(elo_diff)/10 + 70))}%</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # 6. CALCULATEUR DE MISE
            st.subheader("🧮 CALCULATEUR DE MISE INTELLIGENTE")
            
            if st.session_state.bet_manager:
                bankroll = st.session_state.bet_manager.bankroll
                
                col15, col16 = st.columns(2)
                
                with col15:
                    selected_market = st.selectbox(
                        "Sélectionnez un marché",
                        [f"{bet['sélection']} @ {bet['cote']:.2f}" for bet in value_bets] if value_bets else ["Aucun value bet"]
                    )
                    
                    if value_bets and selected_market != "Aucun value bet":
                        selected_bet = next(bet for bet in value_bets if f"{bet['sélection']} @ {bet['cote']:.2f}" == selected_market)
                        
                        edge = float(selected_bet['edge'].replace('%', '')) / 100
                        odds = selected_bet['cote']
                        
                        # Calcul Kelly
                        kelly_fraction = 0.25
                        kelly_stake = st.session_state.value_detector.calculate_kelly_stake(
                            edge, odds, bankroll, kelly_fraction
                        )
                        
                        with col16:
                            st.metric("Bankroll disponible", f"€{bankroll:,.2f}")
                            st.metric("Mise Kelly (25%)", f"€{kelly_stake:,.2f}")
                            st.metric("% du bankroll", f"{(kelly_stake/bankroll*100):.2f}%")
                            
                            if kelly_stake > 0:
                                if st.button(f"💰 PLACER CE PARI (€{kelly_stake:,.2f})", type="primary"):
                                    match_info = {
                                        'match': f"{home_team} vs {away_team}",
                                        'league': f"{home_league} / {away_league}"
                                    }
                                    
                                    bet_details = {
                                        'market': selected_bet['marché'],
                                        'selection': selected_bet['sélection'],
                                        'probability': float(selected_bet['probabilité']),
                                        'edge': edge
                                    }
                                    
                                    result = st.session_state.bet_manager.place_bet(
                                        match_info, bet_details, kelly_stake, odds
                                    )
                                    
                                    if result['success']:
                                        st.success("✅ Pari placé avec succès !")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Erreur: {result.get('error')}")
    
    # Section d'explication
    st.divider()
    
    st.markdown("""
    ### 📚 Comment utiliser cette analyse:
    
    1. **Saisie des données**: Entrez les informations les plus précises possibles
    2. **Analyse automatique**: Le système calcule les probabilités basées sur:
       - Ratings Élo avancés
       - Forme récente des équipes
       - Statistiques offensives/défensives
       - Facteurs contextuels (météo, absences, importance)
    3. **Recommandations**: Les value bets sont identifiés automatiquement
    4. **Gestion bankroll**: Calcul des mises optimales avec Kelly fractionnaire
    
    ### ⚠️ Limitations:
    - Les prédictions sont basées sur des modèles statistiques
    - Les cotes du marché sont estimées (à ajuster selon les bookmakers)
    - Le football reste imprévisible (facteurs psychologiques, arbitrage, etc.)
    """)
