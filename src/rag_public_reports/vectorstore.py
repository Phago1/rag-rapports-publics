"""
vectorstore.py
--------------
Crée, charge et interroge le vector store Chroma avec Gemini embeddings.

Chroma stocke sur disque : une fois les embeddings calculés, tu ne repasses
plus par l'API d'embedding (c'est là que tu économises de l'argent !).

Utilisation rapide :
    from rag_public_reports.vectorstore import get_vector_store, add_documents

    vs = get_vector_store()          # charge ou crée la base
    add_documents(vs, chunks)        # ajoute des chunks
    results = vs.similarity_search(query, k=6)
"""

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from .config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings Gemini
# ─────────────────────────────────────────────────────────────────────────────

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Retourne le modèle d'embedding Gemini.
    La clé API est lue depuis la variable d'environnement GOOGLE_API_KEY
    (définie dans ton fichier .env).
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# Vector store
# ─────────────────────────────────────────────────────────────────────────────

def get_vector_store(collection_name: str = "rapports_publics") -> Chroma:
    """
    Charge le vector store depuis le disque (ou le crée s'il n'existe pas encore).

    Le dossier CHROMA_DIR est défini dans config.py.
    La collection regroupe tous tes rapports ensemble.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    count = vector_store._collection.count()
    print(f"🗄️  Vector store chargé — {count} chunks en base")
    return vector_store


def add_documents(vector_store: Chroma, chunks: list[Document]) -> list[str]:
    """
    Ajoute des chunks au vector store et retourne leurs IDs.

    ⚠️  Appelle d'abord is_already_ingested() pour éviter les doublons.
    """
    doc_ids = vector_store.add_documents(documents=chunks)
    print(f"➕ {len(doc_ids)} chunks ajoutés au vector store")
    return doc_ids


def is_already_ingested(vector_store: Chroma, source: str) -> bool:
    """
    Vérifie si un PDF a déjà été ingéré (par son chemin de fichier).
    Évite les doublons dans la base.

    Exemple :
        if not is_already_ingested(vs, "data/mon-rapport.pdf"):
            add_documents(vs, chunks)
    """
    source = str(source)   # ← conversion automatique Path → str
    results = vector_store.get(where={"source": source})
    already_in = len(results["ids"]) > 0
    if already_in:
        print(f"⚠️  Déjà ingéré : {source} ({len(results['ids'])} chunks)")
    return already_in


def search(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K,
    filter_institution: str | None = None,
    filter_year: int | None = None,
    filter_theme: str | None = None,
) -> list[Document]:
    """
    Recherche les chunks les plus pertinents pour une question.

    Filtres optionnels sur les métadonnées :
      filter_institution : ex. "IGF" ou "Cour des comptes"
      filter_year        : ex. 2023
      filter_theme       : ex. "finances publiques"

    Les filtres utilisent la syntaxe Chroma (type MongoDB).
    Plusieurs filtres actifs sont combinés avec un AND implicite.

    Exemple :
        docs = search(vs, "Quelles sont les recommandations sur la dette ?",
                      filter_institution="Cour des comptes", filter_year=2023)
    """
    # Construction du filtre
    conditions = {}
    if filter_institution:
        conditions["institution"] = filter_institution
    if filter_year:
        conditions["year"] = filter_year
    if filter_theme:
        conditions["theme"] = filter_theme

    # Chroma veut un "$and" explicite quand il y a plusieurs conditions
    if len(conditions) > 1:
        where = {"$and": [{k: v} for k, v in conditions.items()]}
    elif len(conditions) == 1:
        where = conditions
    else:
        where = None

    kwargs = {"k": k}
    if where:
        kwargs["filter"] = where

    results = vector_store.similarity_search(query, **kwargs)
    print(f"🔍 {len(results)} chunks récupérés")
    return results
