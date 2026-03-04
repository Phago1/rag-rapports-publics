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
    try:
        return get_vector_store()
    except Exception as e:
        st.error(f"⚠️ Impossible de charger le vectorstore : {e}")
        return None

vs = load_vs()

# ─── Interface ────────────────────────────────────────────────────────────────
st.title("✍️ Rédaction assistée de section")

# Filtres dans la sidebar
with st.sidebar:
    st.header("Filtres")
    institution = st.selectbox("Institution", ["Toutes"] + KNOWN_INSTITUTIONS)
    theme = st.selectbox("Thème", ["Tous"] + KNOWN_THEMES)
    year_options = ["Toutes"] + list(range(2026, 2018, -1))
    year = st.selectbox("Année", year_options)

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
        st.warning("Saisir un titre de section")
    elif not notes:
        st.warning("Saisir des notes de terrain")
    elif vs is None:
        st.error("⚠️ Le vectorstore n'est pas disponible. Réessayez dans quelques instants.")
    else:
        try:
            with st.spinner("Rédaction en cours..."):
                reponse, sources = answer(
                    titre,
                    notes=notes,
                    filter_institution=None if institution == "Toutes" else institution,
                    filter_year=None if year == "Toutes" else int(year),
                    filter_theme=None if theme == "Tous" else theme,
                    mode="redaction",
                    vs=vs,
                )
            st.markdown(reponse)

            with st.expander("📄 Sources mobilisées"):
                for doc in sources:
                    m = doc.metadata
                    st.markdown(
                        f"**{m.get('title', 'N/A')}** "
                        f"({m.get('institution', '?')}, {m.get('year', '?')})"
                        f" — *{m.get('section', 'section inconnue')}*"
                    )
                    st.caption(doc.page_content[:300] + "…")
                    st.divider()

            # Bouton de téléchargement
            st.download_button(
                label="📥 Télécharger en .txt",
                data=reponse,
                file_name=f"{titre[:50].replace(' ', '_')}.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"⚠️ Une erreur est survenue : {e}")
            st.info("Conseil : élargissez les filtres ou reformulez le titre.")
