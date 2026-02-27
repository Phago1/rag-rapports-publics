"""
ingest.py
---------
Charge un PDF, le découpe en chunks et enrichit chaque chunk de métadonnées.

Deux stratégies de chunking :
  - "sections"  : exploite la structure propre à chaque institution (prioritaire)
  - "recursive" : découpage fixe avec chevauchement (fallback robuste)
"""

import re
from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import CHUNK_SIZE, CHUNK_OVERLAP
import unicodedata

# ─────────────────────────────────────────────────────────────────────────────
# Traitement des titres
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Minuscules + suppression des accents pour comparaison souple."""
    return unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode()

# ─────────────────────────────────────────────────────────────────────────────
# Chargement du PDF
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf_as_single_doc(file_path: Path) -> Document:
    """Charge le PDF entier comme un seul bloc de texte."""
    loader = PyPDFLoader(str(file_path), mode="single")
    return loader.load()[0]


def _load_pdf_page_by_page(file_path: Path) -> list[Document]:
    """Charge le PDF page par page (conserve les numéros de page dans les métadonnées)."""
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    # Supprime les pages vides (fréquentes dans les rapports institutionnels)
    return [p for p in pages if p.page_content.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Patterns de détection des titres — par institution
# ─────────────────────────────────────────────────────────────────────────────
#
# Chaque institution a ses propres conventions typographiques.
# On définit un jeu de patterns par institution, plus un jeu générique
# utilisé en fallback (ou quand l'institution n'est pas reconnue).
#
# Structure réelle observée :
#
# COUR DES COMPTES :
#   Niveau 1 : "CHAPITRE I  LA PHASE 2018-2022 DE LA STRATÉGIE..."
#   Niveau 2 : "I - LA PREMIÈRE PHASE DE LA SNIA..."       (MAJUSCULES)
#   Niveau 3 : "A - Les objectifs retenus pour..."          (Majuscule initiale)
#   Sans num  : "SYNTHÈSE", "INTRODUCTION", "RÉCAPITULATIF DES RECOMMANDATIONS"
#
# IGF :
#   Niveau 1 : "1.    LES JETONS CONSTITUENT UNE INNOVATION..."  (MAJUSCULES)
#   Niveau 2 : "1.1.  Les jetons à vocation commerciale..."      (Majuscule initiale)
#   Niveau 3 : "1.1.1. Les JVC se développent..."                (Majuscule initiale)
#   Sans num  : "INTRODUCTION", "SYNTHÈSE"
#
# ─────────────────────────────────────────────────────────────────────────────

# Patterns Cour des comptes
_PATTERNS_COUR_DES_COMPTES = [

    # Niveau 1 : CHAPITRE / PARTIE / TITRE + chiffre romain + titre
    # "CHAPITRE I  LA PHASE 2018-2022..."
    r"^(CHAPITRE|PARTIE|TITRE|SECTION)\s+[IVXLC]+\s+.{3,}",

    # Niveau 2 : chiffre romain + tiret + TITRE EN MAJUSCULES
    # "I - LA PREMIÈRE PHASE DE LA SNIA..."
    # On vérifie que le titre est en majuscules (au moins 2 lettres maj de suite)
    r"^[IVX]{1,4}\s*[\-–]\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{2,}.{5,}",

    # Niveau 3 : lettre + tiret + Titre en majuscule initiale puis minuscules
    # "A - Les objectifs retenus pour la première phase..."
    # La minuscule après la 2e lettre distingue ce niveau du niveau 2
    r"^[A-Z]\s*[\-–]\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûü].{5,}",

    # Titres sans numéro, entièrement en majuscules (4 à 80 chars)
    # "SYNTHÈSE", "INTRODUCTION", "RÉCAPITULATIF DES RECOMMANDATIONS"
    r"^(?:[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]+[\s\-–:]+){2,}[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{2,}$",
]

# Patterns IGF
_PATTERNS_IGF = [

    # Niveau 1 : chiffre seul + TITRE EN MAJUSCULES
    # "1.    LES JETONS CONSTITUENT UNE INNOVATION..."
    # \s{2,} : au moins 2 espaces (l'IGF aligne ses titres avec des espaces)
    r"^\d+\.\s{2,}[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{2}.{5,}",

    # Niveau 2 : x.x. + Titre en majuscule initiale
    # "1.1.  Les jetons à vocation commerciale sont utilisés..."
    r"^\d+\.\d+\.\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûü].{5,}",

    # Niveau 3 : x.x.x. + Titre en majuscule initiale
    # "1.1.1. Les JVC se développent dans le secteur..."
    r"^\d+\.\d+\.\d+\.\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][a-zàâéèêëîïôùûü].{5,}",

    # Titres sans numéro, entièrement en majuscules
    # "INTRODUCTION", "SYNTHÈSE"
    r"^(?:[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]+[\s\-–:]+){2,}[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{2,}$",
]

# Patterns génériques — utilisés pour les autres institutions (CGE, IGAS, etc.)
# ou quand l'institution n'est pas reconnue
# Plus souples, donc plus de risques de faux positifs
_PATTERNS_GENERIQUES = [

    # CHAPITRE / PARTIE / TITRE / SECTION (toutes conventions de séparateur)
    r"^(CHAPITRE|PARTIE|TITRE|SECTION)\s+[0-9IVXLC]+[\s\-–:]+.{3,}",

    # Chiffre romain + tiret ou point
    r"^[IVX]{1,4}[\.\s]*[\-–\.]\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ].{3,}",

    # Lettre + tiret ou point
    r"^[A-Z][\.\s]*[\-–\.]\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ].{3,}",

    # Numérotation décimale (tous niveaux)
    r"^\d+(\.\d+)*\.?\s+[A-ZÀÂÉÈÊËÎÏÔÙÛÜ].{3,}",

    # Titres sans numéro en majuscules
    r"^(?:[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]+[\s\-–:]+){2,}[A-ZÀÂÉÈÊËÎÏÔÙÛÜ]{2,}$",
]

# Table de correspondance institution → patterns
# La clé est en minuscules pour une comparaison souple
_PATTERNS_BY_INSTITUTION = {
    "cour des comptes": _PATTERNS_COUR_DES_COMPTES,
    "igf":              _PATTERNS_IGF,
    "cge":              _PATTERNS_GENERIQUES,   # à affiner quand tu auras un exemple
    "igas":             _PATTERNS_GENERIQUES,   # idem
    "iga":              _PATTERNS_GENERIQUES,   # idem
}

# Sections à ne jamais découper — elles doivent rester entières
# pour que le LLM puisse répondre à des questions comme "toutes les recommandations"
PROTECTED_SECTIONS = [
    "recommandation",   # capture "recommandation" ET "recommandations"
    "conclusion",
    "synthese",         # sans accent — on va normaliser
    "recapitulatif",    # sans accent aussi
]

def _is_protected_section(title: str) -> bool:
    """
    Retourne True si ce titre correspond à une section à protéger.
    Insensible à la casse et aux accents.
    """
    if not title:
        return False

    # Normalisation : minuscules + suppression des accents
    import unicodedata
    def normalize(s):
        s = s.lower()
        s = unicodedata.normalize("NFD", s)           # décompose les caractères accentués
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # supprime les accents
        return s

    title_normalized = normalize(title)

    return any(keyword in title_normalized for keyword in PROTECTED_SECTIONS)


def _get_patterns(institution: str) -> list:
    """
    Retourne les patterns compilés pour une institution donnée.
    Fallback sur les patterns génériques si l'institution n'est pas reconnue.
    """
    # Normalisation : minuscules + suppression des espaces superflus
    key = institution.strip().lower()
    raw_patterns = _PATTERNS_BY_INSTITUTION.get(key, _PATTERNS_GENERIQUES)
    return [re.compile(p, re.MULTILINE) for p in raw_patterns]


def _detect_section_title(
    text: str,
    compiled_patterns: list,
    title: str = "",
) -> str | None:
    """
    Parcourt toutes les lignes non vides du texte pour détecter un titre.
    Retourne le PREMIER titre trouvé (tronqué à 120 chars), ou None.

    Garde-fous appliqués pour éviter les faux positifs :
    - 1 mot de 5 chars ou moins → acronyme ou bruit (ex: "FTAP")
    - Ligne qui finit par un numéro isolé → header de page (ex: "DINUM   41")
    - Ligne répétée dans le texte de la page → header imprimé en en-tête
    - Ligne qui contient le titre du rapport → header récurrent inter-pages
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        for pattern in compiled_patterns:
            if pattern.match(line):
                # Garde-fou 1 : acronyme trop court
                if len(line.split()) == 1 and len(line) <= 5:
                    continue
                # Garde-fou 2 : finit par un numéro de page
                if re.search(r'\s+\d{1,3}\s*$', line):
                    continue
                # Garde-fou 3 : répété dans la même page
                if text.count(line[:40]) > 1:
                    continue
                # Garde-fou 4 : contient le titre du rapport (header inter-pages)
                if title and _normalize(line) in _normalize(title):
                    continue
                return line[:120]
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Stratégie 1 : Chunking par sections
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_by_sections(pages: list[Document], institution: str, title: str = "") -> list[Document]:
    """
    Découpe le document en respectant les titres de sections.
    Utilise les patterns adaptés à l'institution.
    """
    compiled_patterns = _get_patterns(institution)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n\n", "\n\n", "\n", ". ", " "],
    )

    sections: list[Document] = []
    current_text = ""
    current_page_meta = pages[0].metadata.copy() if pages else {}
    current_section_title = None
    section_index = 0

    def _flush_section(text: str, meta: dict, title_: str | None, idx: int):
        text = text.strip()
        if not text:
            return
        doc = Document(
            page_content=text,
            metadata={**meta, "section": title_ or "", "section_index": idx},
        )
        if len(text) > CHUNK_SIZE * 2:
            sub_chunks = recursive_splitter.split_documents([doc])
            for sc in sub_chunks:
                sc.metadata.setdefault("section", title_ or "")
                sc.metadata.setdefault("section_index", idx)
            sections.extend(sub_chunks)
        else:
            sections.append(doc)

    for page in pages:
        # 🆕 title passé au garde-fou 4
        detected = _detect_section_title(page.page_content, compiled_patterns, title)

        if detected and current_text:
            _flush_section(current_text, current_page_meta, current_section_title, section_index)
            section_index += 1
            current_text = page.page_content
            current_section_title = detected
            current_page_meta = page.metadata.copy()
        else:
            current_text += "\n\n" + page.page_content
            if detected and current_section_title is None:
                current_section_title = detected

    _flush_section(current_text, current_page_meta, current_section_title, section_index)

    return sections

# ─────────────────────────────────────────────────────────────────────────────
# Stratégie 2 : Chunking récursif (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_recursive(doc: Document) -> list[Document]:
    """
    Découpe le document en chunks de taille fixe avec chevauchement.
    Simple, robuste, insensible à la qualité du PDF.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n\n", "\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents([doc])


# ─────────────────────────────────────────────────────────────────────────────
# Enrichissement des métadonnées
# ─────────────────────────────────────────────────────────────────────────────

def _add_metadata(
    chunks: list[Document],
    institution: str,
    year: int,
    title: str,
    theme: str,
    file_path: Path,
) -> list[Document]:
    """Enrichit chaque chunk avec les métadonnées du rapport."""
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "institution": institution,
            "year": year,
            "title": title,
            "theme": theme,
            "source": str(file_path),
            "chunk_index": i,
        })
        chunk.metadata.setdefault("section", "")
        chunk.metadata.setdefault("section_index", -1)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf(
    file_path: str | Path,
    institution: str,
    year: int,
    title: str,
    theme: str = "",
    strategy: Literal["sections", "recursive"] = "sections",
) -> list[Document]:
    """
    Charge un PDF et retourne une liste de chunks enrichis de métadonnées.

    Paramètres
    ----------
    file_path   : chemin vers le PDF
    institution : ex. "IGF", "Cour des comptes" — détermine les patterns utilisés
    year        : année du rapport (ex. 2023)
    title       : titre court du rapport
    theme       : thématique principale ex. "numérique"
    strategy    : "sections" (prioritaire) ou "recursive" (fallback)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {file_path}")

    print(f"📄  Chargement : {file_path.name}  [stratégie : {strategy}]")

    if strategy == "sections":
        pages = _load_pdf_page_by_page(file_path)
        print(f"    → {len(pages)} pages chargées")
        chunks = _chunk_by_sections(pages, institution, title=title)  # 🆕 title=title
    else:
        doc = _load_pdf_as_single_doc(file_path)
        chunks = _chunk_recursive(doc)

    # 🆕 Enrichissement des métadonnées — NE PAS OUBLIER
    chunks = _add_metadata(chunks, institution, year, title, theme, file_path)

    # Bilan
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
    sections_detected = sum(1 for c in chunks if c.metadata.get("section"))
    print(f"✅  {len(chunks)} chunks créés")
    print(f"    → Longueur moyenne : {avg_len} caractères")
    if strategy == "sections":
        print(f"    → Sections détectées : {sections_detected} chunks avec titre de section")

    return chunks
