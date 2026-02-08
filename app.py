def display_match_analysis_manual():
    """Analyse manuelle d'un match - Version sans Plotly"""
    st.header("🔍 ANALYSE DE MATCH MANUELLE")
    
    st.info("Entrez les détails d'un match pour obtenir une analyse détaillée avec prédictions et recommandations de paris.")
    
    # Formulaire de saisie simplifié
    with st.form("match_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Équipe Domicile")
            home_team = st.text_input("Nom", "Manchester City")
            home_form = st.slider("Forme (1-10)", 1, 10, 7)
            home_attack = st.number_input("Attaque (buts/moy)", 0.0, 5.0, 2.3, 0.1)
            home_defense = st.number_input("Défense (buts/moy)", 0.0, 5.0, 0.8, 0.1)
        
        with col2:
            st.subheader("⚽ Équipe Extérieur")
            away_team = st.text_input("Nom", "Liverpool")
            away_form = st.slider("Forme (1-10)", 1, 10, 6)
            away_attack = st.number_input("Attaque (buts/moy)", 0.0, 5.0, 1.9, 0.1)
            away_defense = st.number_input("Défense (buts/moy)", 0.0, 5.0, 1.2, 0.1)
        
        # Paramètres supplémentaires
        st.subheader("⚙️ Paramètres supplémentaires")
        col3, col4 = st.columns(2)
        
        with col3:
            is_neutral = st.checkbox("Terrain neutre")
            importance = st.selectbox("Importance", ["Normal", "Coupe", "Dernière journée", "Finale"])
        
        with col4:
            weather = st.selectbox("Météo", ["Bonnes", "Pluie", "Vent", "Froid", "Chaud"])
            home_missing = st.number_input("Absents domicile", 0, 10, 1)
            away_missing = st.number_input("Absents extérieur", 0, 10, 2)
        
        # Bouton d'analyse
        submitted = st.form_submit_button("🚀 ANALYSER LE MATCH", type="primary")
    
    if submitted:
        try:
            # 1. CALCUL DES RATINGS
            st.subheader("📈 RATINGS DES ÉQUIPES")
            
            # Calcul simplifié des ratings
            home_rating = 1500 + (home_form - 5) * 50 + (home_attack - away_defense) * 100
            away_rating = 1500 + (away_form - 5) * 50 + (away_attack - home_defense) * 100
            
            # Ajustements
            if not is_neutral:
                home_rating += 70
            
            if importance in ["Finale", "Dernière journée"]:
                home_rating *= 1.1
                away_rating *= 1.1
            
            col5, col6 = st.columns(2)
            
            with col5:
                st.metric(f"🏠 {home_team}", f"{home_rating:.0f}")
                
                # Graphique simple avec barres natives Streamlit
                st.write("**Profil de l'équipe:**")
                st.write(f"• Forme: {home_form}/10")
                st.write(f"• Attaque: {home_attack} buts/moy")
                st.write(f"• Défense: {home_defense} buts/moy")
                
                # Barres de progression
                st.progress(home_form / 10, text="Forme")
                st.progress(min(home_attack / 3, 1.0), text="Attaque")
                st.progress(max(0, 1 - home_defense / 3), text="Défense")
            
            with col6:
                st.metric(f"⚽ {away_team}", f"{away_rating:.0f}")
                
                st.write("**Profil de l'équipe:**")
                st.write(f"• Forme: {away_form}/10")
                st.write(f"• Attaque: {away_attack} buts/moy")
                st.write(f"• Défense: {away_defense} buts/moy")
                
                # Barres de progression
                st.progress(away_form / 10, text="Forme")
                st.progress(min(away_attack / 3, 1.0), text="Attaque")
                st.progress(max(0, 1 - away_defense / 3), text="Défense")
            
            # 2. PRÉDICTIONS
            st.subheader("🎯 PRÉDICTIONS DU MATCH")
            
            # Calcul des probabilités
            rating_diff = home_rating - away_rating
            home_win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
            draw_prob = 0.3 * np.exp(-abs(rating_diff) / 300)
            away_win_prob = 1 - home_win_prob - draw_prob
            
            # Afficher en colonnes
            col7, col8, col9 = st.columns(3)
            
            with col7:
                st.markdown(f"""
                <div style="background: #E3F2FD; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>🏠 {home_team}</h4>
                <h2 style="color: #1E88E5;">{home_win_prob*100:.1f}%</h2>
                <p>Cote: {1/home_win_prob:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                st.markdown(f"""
                <div style="background: #F3E5F5; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>🤝 NUL</h4>
                <h2 style="color: #9C27B0;">{draw_prob*100:.1f}%</h2>
                <p>Cote: {1/draw_prob:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col9:
                st.markdown(f"""
                <div style="background: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>⚽ {away_team}</h4>
                <h2 style="color: #4CAF50;">{away_win_prob*100:.1f}%</h2>
                <p>Cote: {1/away_win_prob:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 3. SCORE PRÉDIT
            st.subheader("📊 SCORE LE PLUS PROBABLE")
            
            # Buts attendus
            expected_home = (home_attack + away_defense) / 2
            expected_away = (away_attack + home_defense) / 2
            
            # Ajustements météo
            if weather != "Bonnes":
                expected_home *= 0.9
                expected_away *= 0.9
            
            # Score le plus probable
            predicted_home = round(expected_home)
            predicted_away = round(expected_away)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; text-align: center; color: white;">
            <h1 style="font-size: 4rem; margin: 0;">{predicted_home} - {predicted_away}</h1>
            <p style="font-size: 1.2rem;">Score le plus probable</p>
            <p>Buts attendus: {expected_home:.2f} - {expected_away:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. RECOMMANDATIONS
            st.subheader("💰 RECOMMANDATIONS DE PARIS")
            
            # Cotes du marché estimées
            market_odds = {
                'home': 1/home_win_prob * 0.9,  # 10% de marge bookmaker
                'draw': 1/draw_prob * 0.9,
                'away': 1/away_win_prob * 0.9
            }
            
            # Calculer les value bets
            recommendations = []
            
            # Vérifier chaque résultat
            if home_win_prob > 0.5 and market_odds['home'] > 2.0:
                edge = (home_win_prob * market_odds['home']) - 1
                if edge > 0.02:
                    recommendations.append({
                        'type': 'VICTOIRE DOMICILE',
                        'équipe': home_team,
                        'cote': f"{market_odds['home']:.2f}",
                        'probabilité': f"{home_win_prob*100:.1f}%",
                        'edge': f"{edge*100:.1f}%",
                        'niveau': '✅ BONNE'
                    })
            
            if draw_prob > 0.25 and market_odds['draw'] > 3.0:
                edge = (draw_prob * market_odds['draw']) - 1
                if edge > 0.02:
                    recommendations.append({
                        'type': 'MATCH NUL',
                        'équipe': 'Nul',
                        'cote': f"{market_odds['draw']:.2f}",
                        'probabilité': f"{draw_prob*100:.1f}%",
                        'edge': f"{edge*100:.1f}%",
                        'niveau': '⚠️ MODÉRÉE'
                    })
            
            if away_win_prob > 0.3 and market_odds['away'] > 3.5:
                edge = (away_win_prob * market_odds['away']) - 1
                if edge > 0.02:
                    recommendations.append({
                        'type': 'VICTOIRE EXTÉRIEUR',
                        'équipe': away_team,
                        'cote': f"{market_odds['away']:.2f}",
                        'probabilité': f"{away_win_prob*100:.1f}%",
                        'edge': f"{edge*100:.1f}%",
                        'niveau': '🎯 EXCELLENTE'
                    })
            
            # Afficher les recommandations
            if recommendations:
                for rec in recommendations:
                    with st.expander(f"{rec['type']} - {rec['niveau']}"):
                        st.write(f"**Équipe:** {rec['équipe']}")
                        st.write(f"**Cote estimée:** {rec['cote']}")
                        st.write(f"**Probabilité modèle:** {rec['probabilité']}")
                        st.write(f"**Edge (avantage):** {rec['edge']}")
                        
                        # Calcul de mise si bankroll disponible
                        if 'bet_manager' in st.session_state and st.session_state.bet_manager:
                            bankroll = st.session_state.bet_manager.bankroll
                            edge_value = float(rec['edge'].replace('%', '')) / 100
                            odds_value = float(rec['cote'])
                            
                            try:
                                stake = bankroll * edge_value * 0.1  # 10% de l'edge
                                stake = min(stake, bankroll * 0.05)  # Max 5% du bankroll
                                
                                if stake > 10:  # Minimum 10€
                                    st.write(f"**Mise recommandée:** €{stake:.2f}")
                                    
                                    if st.button(f"Placer pari ({rec['type']})", key=f"bet_{rec['type']}"):
                                        match_info = {
                                            'match': f"{home_team} vs {away_team}",
                                            'league': "Analyse manuelle"
                                        }
                                        
                                        bet_details = {
                                            'market': '1X2',
                                            'selection': rec['type'],
                                            'probability': float(rec['probabilité'].replace('%', '')) / 100,
                                            'edge': edge_value
                                        }
                                        
                                        result = st.session_state.bet_manager.place_bet(
                                            match_info, bet_details, stake, odds_value
                                        )
                                        
                                        if result['success']:
                                            st.success(f"✅ Pari placé: €{stake:.2f}")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Erreur: {result.get('error')}")
                            except:
                                pass
            else:
                st.warning("Aucune opportunité de value bet significative détectée.")
            
            # 5. RÉSUMÉ
            st.subheader("📋 RÉSUMÉ DE L'ANALYSE")
            
            col10, col11 = st.columns(2)
            
            with col10:
                st.markdown(f"""
                <div style="background: #FFF3CD; padding: 15px; border-radius: 10px;">
                <h4>📈 AVANTAGES {home_team}:</h4>
                • Rating supérieur: {home_rating:.0f} vs {away_rating:.0f}<br>
                • Forme: {home_form}/10 vs {away_form}/10<br>
                • Avantage terrain: {"Oui" if not is_neutral else "Non"}<br>
                • Derniers matchs: {home_form*2} pts sur {home_form*3} possibles
                </div>
                """, unsafe_allow_html=True)
            
            with col11:
                st.markdown(f"""
                <div style="background: #D1ECF1; padding: 15px; border-radius: 10px;">
                <h4>⚠️ FACTEURS À CONSIDÉRER:</h4>
                • Météo: {weather}<br>
                • Importance: {importance}<br>
                • Absents: {home_missing} vs {away_missing}<br>
                • Confiance: {min(95, int(abs(rating_diff)/20 + 70))}%
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'analyse: {str(e)}")
            st.info("Veuillez vérifier les valeurs saisies et réessayer.")
    
    # Informations supplémentaires
    st.divider()
    st.markdown("""
    ### 📖 Comment utiliser cette analyse:
    
    1. **Saisissez les données** des deux équipes (forme, attaque, défense)
    2. **Ajustez les paramètres** contextuels (météo, importance, absences)
    3. **Cliquez sur ANALYSER** pour obtenir les prédictions
    4. **Consultez les recommandations** de paris avec calculs d'edge
    5. **Utilisez le calculateur** de mise pour optimiser vos paris
    
    ### ⚠️ Note importante:
    Les prédictions sont basées sur des modèles statistiques et ne garantissent pas les résultats.
    """)
