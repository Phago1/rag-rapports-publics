"""app.py — Interface principale du RAG rapports publics."""
import streamlit as st
from rag_public_reports.vectorstore import get_vector_store
from rag_public_reports.rag import answer
from rag_public_reports.config import KNOWN_INSTITUTIONS, KNOWN_THEMES

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="Recherche rapports publics - appli PB",
    page_icon="📋",
    layout="wide",
)

# ─── Chargement du vector store (une seule fois) ─────────────────────────────
@st.cache_resource
def load_vs():
    return get_vector_store()

vs = load_vs()

# ─── Interface ───────────────────────────────────────────────────────────────
st.title("📋 Appli recherche rapports publics")

# Filtres dans la sidebar
with st.sidebar:
    st.header("Filtres")
    institution = st.selectbox("Institution", ["Toutes"] + KNOWN_INSTITUTIONS)
    theme = st.selectbox("Thème", ["Tous"] + KNOWN_THEMES)
    mode = st.selectbox("Mode", ["rag", "synthesis"])

# Zone de question
question = st.text_input("Votre question", placeholder="Ex : Quelles sont les recommandations sur l'IA ?")

if st.button("Envoi", type="primary"):
    if not question:
        st.warning("Saisissez une question")
    else:
        with st.spinner("Recherche en cours..."):
            reponse = answer(
                question,
                filter_institution=None if institution == "Toutes" else institution,
                filter_theme=None if theme == "Tous" else theme,
                mode=mode,
                vs=vs,
            )
        st.markdown(reponse)
