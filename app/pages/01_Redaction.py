"""01_Redaction.py — Page de rédaction assistée."""
import streamlit as st
from rag_public_reports.vectorstore import get_vector_store
from rag_public_reports.rag import answer
from rag_public_reports.config import KNOWN_INSTITUTIONS, KNOWN_THEMES

# ─── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rédaction assistée",
    page_icon="✍️",
    layout="wide",
)

# ─── Chargement du vector store (une seule fois) ──────────────────────────────
@st.cache_resource
def load_vs():
    return get_vector_store()

vs = load_vs()

# ─── Interface ────────────────────────────────────────────────────────────────
st.title("✍️ Rédaction assistée de section")

# Filtres dans la sidebar
with st.sidebar:
    st.header("Filtres")
    institution = st.selectbox("Institution", ["Toutes"] + KNOWN_INSTITUTIONS)
    theme = st.selectbox("Thème", ["Tous"] + KNOWN_THEMES)

# Inputs utilisateur
titre = st.text_input(
    "Titre de la section",
    placeholder="Ex : La transformation de l'action publique par l'IA"
)

notes = st.text_area(
    "Notes de terrain",
    placeholder="Colle ici tes observations, chiffres, constats issus de l'instruction...",
    height=250,
)

if st.button("Rédiger", type="primary"):
    if not titre:
        st.warning("Saisis un titre de section !")
    elif not notes:
        st.warning("Saisis des notes de terrain !")
    else:
        with st.spinner("Rédaction en cours..."):
            reponse = answer(
                titre,
                notes=notes,
                filter_institution=None if institution == "Toutes" else institution,
                filter_theme=None if theme == "Tous" else theme,
                mode="redaction",
                vs=vs,
            )
        st.markdown(reponse)

        # Bouton de copie
        st.download_button(
            label="📥 Télécharger en .txt",
            data=reponse,
            file_name=f"{titre[:50].replace(' ', '_')}.txt",
            mime="text/plain",
        )
