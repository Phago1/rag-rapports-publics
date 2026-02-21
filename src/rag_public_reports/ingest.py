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
    r"^[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][A-ZÀÂÉÈÊËÎÏÔÙÛÜ\s\-–:]{3,79}$",
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
    r"^[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][A-ZÀÂÉÈÊËÎÏÔÙÛÜ\s\-–:]{3,79}$",
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
    r"^[A-ZÀÂÉÈÊËÎÏÔÙÛÜ][A-ZÀÂÉÈÊËÎÏÔÙÛÜ\s\-–:]{3,79}$",
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


def _detect_section_title(text: str, compiled_patterns: list) -> str | None:
    """
    Parcourt toutes les lignes non vides du texte pour détecter un titre.
    Retourne le PREMIER titre trouvé (tronqué à 120 chars), ou None.

    Pourquoi toutes les lignes et pas seulement les premières ?
    Dans les rapports institutionnels, un titre de section peut apparaître
    n'importe où dans la page — pas seulement en haut. Par exemple un
    sous-titre de niveau 2 ou 3 peut se trouver au milieu d'une page
    après le texte de la section précédente.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        for pattern in compiled_patterns:
            if pattern.match(line):
                return line[:120]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stratégie 1 : Chunking par sections
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_by_sections(doc: Document, institution: str) -> list[Document]:
    """
    Découpe le document complet en respectant les titres de sections.
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
    current_section_title = None
    section_index = 0

    lines = doc.page_content.split("\n")

    def _flush_section(text: str, title: str | None, idx: int):
        text = text.strip()
        if not text:
            return
        chunk = Document(
            page_content=text,
            metadata={
                **doc.metadata,          # conserve source, etc.
                "section": title or "",
                "section_index": idx,
            },
        )
        if len(text) > CHUNK_SIZE * 2:
            sub_chunks = recursive_splitter.split_documents([chunk])
            for sc in sub_chunks:
                sc.metadata.setdefault("section", title or "")
                sc.metadata.setdefault("section_index", idx)
            sections.extend(sub_chunks)
        else:
            sections.append(chunk)

    for line in lines:
        line_stripped = re.sub(r"\s+", " ", line).strip()  # nettoyage espaces parasites
        detected = _detect_section_title(line_stripped, compiled_patterns)

        if detected and current_text:
            _flush_section(current_text, current_section_title, section_index)
            section_index += 1
            current_text = line
            current_section_title = detected
        else:
            current_text += "\n" + line

    _flush_section(current_text, current_section_title, section_index)

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

    doc = _load_pdf_as_single_doc(file_path)
    print(f"    → {len(doc.page_content)} caractères chargés")

    if strategy == "sections":
        chunks = _chunk_by_sections(doc, institution)
    else:
        chunks = _chunk_recursive(doc)

    chunks = _add_metadata(chunks, institution, year, title, theme, file_path)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 150]
    
    # Bilan
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
    sections_detected = sum(1 for c in chunks if c.metadata.get("section"))
    print(f"✅  {len(chunks)} chunks créés")
    print(f"    → Longueur moyenne : {avg_len} caractères")
    if strategy == "sections":
        print(f"    → Sections détectées : {sections_detected} chunks avec titre de section")

    return chunks
