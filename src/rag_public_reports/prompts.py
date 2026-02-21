"""
prompts.py
----------
Templates de prompts adaptés aux rapports institutionnels français.

Pourquoi des prompts personnalisés ?
  - Les rapports IGF / Cour des comptes ont un style très particulier
  - On veut que le LLM cite ses sources (numéro de page, institution, année)
  - On veut éviter les hallucinations en forçant le LLM à se baser sur les chunks

Utilisation :
    from rag_public_reports.prompts import get_rag_prompt
    prompt = get_rag_prompt()
"""

from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────────────────────────────────────────
# Prompt principal RAG
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """Tu es un assistant expert en rapports institutionnels français \
(Cour des comptes, IGF, CGE, IGAS, IGA, etc.).

Tu as accès à des extraits de rapports officiels. Réponds à la question de l'utilisateur \
en te basant UNIQUEMENT sur ces extraits.

Règles importantes :
1. Si l'information n'est pas dans les extraits, dis-le clairement — ne pas inventer.
2. Cite tes sources : mentionne l'institution, l'année et, si disponible, le titre de section.
3. Sois précis et concis. Utilise le style factuel des rapports administratifs.
4. Si plusieurs rapports donnent des informations contradictoires, mentionne-le.

Extraits disponibles :
{context}
"""

RAG_HUMAN_PROMPT = "Question : {question}"


def get_rag_prompt() -> ChatPromptTemplate:
    """Retourne le prompt RAG principal."""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_HUMAN_PROMPT),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Prompt de synthèse (pour comparer plusieurs rapports)
# ─────────────────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """Tu es un assistant expert en politiques publiques françaises.

Tu dois synthétiser ce que disent plusieurs rapports institutionnels sur un même sujet.

Structure ta réponse ainsi :
1. **Principaux constats** : ce sur quoi les rapports s'accordent
2. **Nuances et divergences** : points de désaccord ou d'évolution dans le temps
3. **Recommandations clés** : les recommandations les plus importantes

Base-toi UNIQUEMENT sur les extraits fournis. Cite les sources.

Extraits :
{context}
"""

SYNTHESIS_HUMAN_PROMPT = "Sujet à synthétiser : {question}"


def get_synthesis_prompt() -> ChatPromptTemplate:
    """Retourne le prompt de synthèse multi-rapports."""
    return ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_SYSTEM_PROMPT),
        ("human", SYNTHESIS_HUMAN_PROMPT),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Prompt de rédaction de section
# ─────────────────────────────────────────────────────────────────────────────

REDACTION_SYSTEM_PROMPT = """Tu es un expert en rédaction de rapports institutionnels français, \
spécialisé dans le style de la Cour des comptes et de l'IGF.

Tu dois rédiger une section de rapport en articulant trois sources :
1. Le titre de section (si proposé par le rédacteur) et les notes du rédacteur (prioritaires)
2. Les extraits de rapports existants (pour le contexte et les références)
3. Tes connaissances générales en complément — mais tu dois alors le signaler explicitement \
par la mention "(source : externe)"

Structure attendue :
- Un titre de section (repris ou reformulé à partir du titre fourni)
- Deux ou trois sous-sections avec sous-titres si le contenu des notes du rédacteur \
le justifie
- entre 600 et 800 mots au total (marge de plus ou moins 10%)
- Ton factuel, administratif et précis — style Cour des comptes et IGF
- Chiffres et données sourcés

Notes du rédacteur :
{notes}

Extraits de rapports existants :
{context}
"""

REDACTION_HUMAN_PROMPT = "Rédige une section de rapport intitulée : {titre}"


def get_redaction_prompt() -> ChatPromptTemplate:
    """Retourne le prompt de rédaction de section."""
    return ChatPromptTemplate.from_messages([
        ("system", REDACTION_SYSTEM_PROMPT),
        ("human", REDACTION_HUMAN_PROMPT),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Formatage du contexte (les chunks récupérés)
# ─────────────────────────────────────────────────────────────────────────────

def format_context(docs) -> str:
    """
    Met en forme les chunks récupérés pour les injecter dans le prompt.
    Chaque chunk est présenté avec ses métadonnées pour que le LLM puisse citer.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[Extrait {i}]"

        # On construit une ligne source lisible
        source_parts = []
        if meta.get("institution"):
            source_parts.append(meta["institution"])
        if meta.get("title"):
            source_parts.append(f'"{meta["title"]}"')
        if meta.get("year"):
            source_parts.append(f"({meta['year']})")
        if meta.get("section"):
            source_parts.append(f"— Section : {meta['section']}")
        if meta.get("page"):
            source_parts.append(f"p. {meta['page']}")

        source_line = " ".join(source_parts) if source_parts else "Source inconnue"

        parts.append(f"{header}\nSource : {source_line}\n\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)
