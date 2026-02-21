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
from .prompts import get_rag_prompt, get_synthesis_prompt, get_redaction_prompt,\
    format_context


def _get_llm():
    return init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER)


def answer(
    query: str,
    notes : str = "",
    filter_institution: str | None = None,
    filter_year: int | None = None,
    filter_theme: str | None = None,
    k: int = TOP_K,
    mode: str = "rag",          # "rag" ou "synthesis"
    verbose: bool = False,
    vs=None,
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
    mode               : "rag" (réponse factuelle), "synthesis" (synthèse multi-rapports)\
    ou "redaction"
    verbose            : affiche les chunks récupérés pour vérifier la qualité du retrieval
    """
    # Charger le vector store
    if vs is None:
        vs = get_vector_store()

    # Query fusionnée avec les notes de l'utilisateur
    search_query = f"{query}\n{notes}" if notes else query

    # Chercher les chunks pertinents
    docs = search(
        vs, search_query, k=k,
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

    # Affichage debug (optionnel)
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

    # Mise en forme du contexte
    context = format_context(docs)

    # Choix du prompt et des variables selon le mode
    prompt_configs = {
        "synthesis": (get_synthesis_prompt(), {"context": context, "question": query}),
        "redaction": (get_redaction_prompt(), {"context": context, "titre": query, "notes": notes or "Aucune note fournie."}),
        "rag":       (get_rag_prompt(),       {"context": context, "question": query}),
    }

    prompt_template, variables = prompt_configs.get(mode, prompt_configs["rag"])
    prompt = prompt_template.invoke(variables)

    # Appel au LLM Gemini
    llm = _get_llm()
    response = llm.invoke(prompt)

    return response.content
