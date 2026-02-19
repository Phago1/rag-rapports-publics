"""
utils.py
--------
Fonctions utilitaires pour explorer et gérer ta base de rapports.
Très utile depuis un notebook pour voir ce qui est déjà indexé.
"""

from collections import Counter
from langchain_chroma import Chroma
from .vectorstore import get_vector_store


def list_ingested_reports(vector_store: Chroma | None = None) -> list[dict]:
    """
    Liste tous les rapports déjà indexés dans le vector store.
    Retourne une liste de dicts avec institution, year, title, nb_chunks.

    Exemple :
        from rag_public_reports.utils import list_ingested_reports
        list_ingested_reports()
    """
    if vector_store is None:
        vector_store = get_vector_store()

    # Récupère toutes les métadonnées (sans les vecteurs ni les textes)
    all_meta = vector_store.get()["metadatas"]

    if not all_meta:
        print("📭 La base est vide. Lance ingest_pdf() pour ajouter des rapports.")
        return []

    # Regroupe par (institution, year, title)
    report_keys = [
        (m.get("institution", "?"), m.get("year", "?"), m.get("title", "?"))
        for m in all_meta
    ]
    counts = Counter(report_keys)

    reports = []
    for (institution, year, title), nb_chunks in sorted(counts.items()):
        reports.append({
            "institution": institution,
            "year": year,
            "title": title,
            "nb_chunks": nb_chunks,
        })

    # Affichage
    print(f"\n📚 {len(reports)} rapport(s) indexé(s) — {sum(r['nb_chunks'] for r in reports)} chunks au total\n")
    print(f"{'Institution':<20} {'Année':<8} {'Chunks':<8} Titre")
    print("-" * 80)
    for r in reports:
        print(f"{r['institution']:<20} {str(r['year']):<8} {r['nb_chunks']:<8} {r['title']}")

    return reports


def print_chunk_sample(query: str, vector_store: Chroma | None = None, k: int = 3):
    """
    Affiche les chunks récupérés pour une question (utile pour déboguer le retrieval).

    Exemple :
        print_chunk_sample("recommandations sur la dette")
    """
    if vector_store is None:
        vector_store = get_vector_store()

    docs = vector_store.similarity_search(query, k=k)

    print(f"\n🔍 Top {k} chunks pour : \"{query}\"\n")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"─── Chunk {i} ───────────────────────────────────────────")
        print(f"Source : {meta.get('institution','')} {meta.get('year','')} — {meta.get('title','')}")
        if meta.get("section"):
            print(f"Section : {meta['section']}")
        print(f"\n{doc.page_content[:500]}")
        print()
