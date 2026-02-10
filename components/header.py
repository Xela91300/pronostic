import streamlit as st

def render_header():
    """Affiche l'en-tête de l'application"""
    st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #38a9ff, #0066cc); border-radius: 10px; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0;">⚽ Football Betting Analytics</h1>
            <p style="color: white; opacity: 0.9; margin-top: 10px;">Outils d'analyse automatisés pour les paris sportifs</p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Affiche la sidebar avec navigation"""
    with st.sidebar:
        st.markdown("## Navigation")
        
        # Sélection de la page
        page = st.radio(
            "Menu principal",
            ["Dashboard", "Prédictions CIA", "Statistiques", "Top 50", "Alertes", "Historique"]
        )
        
        # Options de configuration
        st.markdown("---")
        st.markdown("### Configuration")
        
        timezone = st.selectbox(
            "Fuseau horaire",
            ["Europe/Paris", "Europe/London", "America/New_York", "Asia/Tokyo", "Australia/Sydney"],
            index=0
        )
        
        language = st.selectbox(
            "Langue",
            ["Français", "English", "Español"],
            index=0
        )
        
        if st.button("Appliquer"):
            st.session_state.timezone = timezone
            st.session_state.language = language
            st.success("Configuration mise à jour!")
            st.rerun()
        
        return page
