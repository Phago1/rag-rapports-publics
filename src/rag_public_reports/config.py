"""
config.py
---------
Tous les paramètres du projet en un seul endroit.
Si tu veux changer un réglage, c'est ici que ça se passe !
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Chargement du .env ────────────────────────────────────────────────────────
# load_dotenv() lit le fichier .env à la racine du repo et injecte les variables
# dans os.environ — comme ça, tout le code y a accès via os.environ.get(...)
load_dotenv()


# ── Chemins ───────────────────────────────────────────────────────────────────
# Path(__file__) = chemin de ce fichier config.py
# .resolve()     = chemin absolu (pas de "../" relatifs)
# .parents[2]    = remonte 2 niveaux → racine du repo
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"

# On lit VECTORSTORE_DIR depuis le .env si défini, sinon valeur par défaut.
# ROOT_DIR / "data/vectorstore" → chemin absolu complet
_vs_relative = os.environ.get("VECTORSTORE_DIR", "data/vectorstore")
CHROMA_DIR = ROOT_DIR / _vs_relative


# ── Modèles ───────────────────────────────────────────────────────────────────
# Embeddings : Gemini transforme le texte en vecteurs numériques.
# La clé API est lue automatiquement depuis GOOGLE_API_KEY dans le .env.
EMBEDDING_PROVIDER = "google_genai"
EMBEDDING_MODEL    = "models/gemini-embedding-001"

# LLM : Claude génère les réponses à partir des chunks récupérés.
# La clé API est lue automatiquement depuis ANTHROPIC_API_KEY dans le .env.
LLM_PROVIDER = "anthropic"
LLM_MODEL    = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
#                              ↑ on lit depuis le .env, avec une valeur par défaut
#                                au cas où la variable serait absente


# ── Chunking ──────────────────────────────────────────────────────────────────
# Gemini embedding accepte jusqu'à ~2 000 tokens (~8 000 caractères).
# On reste confortablement en dessous avec 2 000 caractères (~400 mots, ~1/2 page A4).
CHUNK_SIZE    = 2_000

# Chevauchement entre deux chunks consécutifs.
# Évite de couper un raisonnement en deux : la fin du chunk N
# est répétée au début du chunk N+1.
CHUNK_OVERLAP = 400


# ── Retrieval ─────────────────────────────────────────────────────────────────
# Nombre de chunks renvoyés par la similarity search.
# 6 est un bon compromis : assez de contexte sans surcharger le prompt de Claude.
TOP_K = 10


# ── Référentiels métier ───────────────────────────────────────────────────────
# Ces listes servent à valider / suggérer les métadonnées lors de l'ingestion.
KNOWN_INSTITUTIONS = [
    "Cour des comptes",
    "IGF",    # Inspection générale des finances
    "CGE",    # Conseil général de l'économie
    "IGAS",   # Inspection générale des affaires sociales
    "IGA",    # Inspection générale de l'administration
    "CGEDD",  # Conseil général de l'environnement
    "Sénat",
    "Assemblée nationale",
]

KNOWN_THEMES = [
    "finances publiques",
    "environnement",
    "santé",
    "éducation",
    "emploi et travail",
    "logement",
    "sécurité sociale",
    "collectivités territoriales",
    "IA et numérique",
    "intérieur et défense",
    "justice",
]
