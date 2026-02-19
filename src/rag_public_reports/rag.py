"""
rag.py
------
La fonction answer() orchestre la recherche + la génération.
Elle accepte maintenant un filtre sur la thématique.

Utilisation :
    from rag_public_reports.rag import answer

    # Question simple (tous les rapports)
    print(answer("Quelles sont les recommandations sur la dette ?"))

    # Filtrée par institution + année
    print(answer("...", filter_institution="IGF", filter_year=2023))

    # Filtrée par thématique
    print(answer("...", filter_theme="finances publiques"))

    # Synthèse multi-rapports
    print(answer("...", mode="synthesis"))
"""

from langchain.chat_models import init_chat_model

from .config import LLM_PROVIDER, LLM_MODEL, TOP_K
from .vectorstore import get_vector_store, search
from .prompts import get_rag_prompt, get_synthesis_prompt, format_context


def _get_llm():
    return init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER)


def answer(
    query: str,
    filter_institution: str | None = None,
    filter_year: int | None = None,
    filter_theme: str | None = None,
    k: int = TOP_K,
    mode: str = "rag",          # "rag" ou "synthesis"
    verbose: bool = False,
) -> str:
    """
    Répond à une question en cherchant dans les rapports indexés.

    Paramètres
    ----------
    query              : question en langage naturel
    filter_institution : ex. "IGF", "Cour des comptes"
    filter_year        : ex. 2023
    filter_theme       : ex. "finances publiques"
    k                  : nombre de chunks à récupérer
    mode               : "rag" (réponse factuelle) ou "synthesis" (synthèse multi-rapports)
    verbose            : affiche les chunks récupérés pour vérifier la qualité du retrieval
    """
    # 1. Charger le vector store
    vs = get_vector_store()

    # 2. Chercher les chunks pertinents
    docs = search(
        vs, query, k=k,
        filter_institution=filter_institution,
        filter_year=filter_year,
        filter_theme=filter_theme,
    )

    if not docs:
        return (
            "❌ Aucun document trouvé pour ces critères.\n"
            "Conseil : élargis les filtres ou vérifie que des rapports sont bien indexés "
            "(utils.list_ingested_reports())."
        )

    # 3. Affichage debug (optionnel)
    if verbose:
        print("\n" + "=" * 60)
        print("CHUNKS RÉCUPÉRÉS :")
        print("=" * 60)
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            print(f"\n[{i}] {m.get('institution','')} {m.get('year','')} — "
                  f"{m.get('section', 'section inconnue')[:80]}")
            print(doc.page_content[:300] + "…")
        print("=" * 60 + "\n")

    # 4. Mise en forme du contexte
    context = format_context(docs)

    # 5. Choix du prompt
    prompt_template = get_synthesis_prompt() if mode == "synthesis" else get_rag_prompt()
    prompt = prompt_template.invoke({"context": context, "question": query})

    # 6. Appel au LLM Gemini
    llm = _get_llm()
    response = llm.invoke(prompt)

    return response.content
