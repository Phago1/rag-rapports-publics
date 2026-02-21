import csv
import json
import os
from pathlib import Path
from rag_public_reports.config import KNOWN_INSTITUTIONS, KNOWN_THEMES
import anthropic
from langchain_community.document_loaders import PyPDFLoader

# Client Anthropic (lit ANTHROPIC_API_KEY depuis .env)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def extraire_metadata(pdf_path: str) -> dict:
    """Lit un PDF et demande à Claude d'extraire les métadonnées."""
    # Lire les premières pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    extrait = "\n\n".join([p.page_content for p in pages[:3]])

    # Appel Claude
    prompt = f"""Tu es un assistant qui extrait des métadonnées de rapports publics français.

Voici le début d'un rapport :

{extrait[:3000]}

Contraintes STRICTES :
- "institution" : choisis UNE SEULE valeur parmi : {KNOWN_INSTITUTIONS}
- "theme" : choisis UNE SEULE valeur parmi : {KNOWN_THEMES}
- "year" : année de publication (entier)
- "title" : titre complet du rapport

Réponds UNIQUEMENT en JSON valide, sans texte autour :
{{
    "title": "...",
    "institution": "...",
    "year": 2024,
    "theme": "..."
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    metadata = json.loads(raw.strip())
    metadata["fichier"] = Path(pdf_path).name
    return metadata


def ajouter_au_catalogue(metadata: dict, catalogue_path: str = "catalogue.csv"):
    """Ajoute une ligne au catalogue.csv si le fichier n'est pas déjà présent."""
    catalogue = Path(catalogue_path)

    # Lire les fichiers déjà présents
    existants = set()
    if catalogue.exists():
        with open(catalogue, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existants = {row["fichier"] for row in reader}

    if metadata["fichier"] in existants:
        print(f"⚠️  Déjà dans le catalogue : {metadata['fichier']}")
        return

    # Ajouter la ligne
    with open(catalogue, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fichier","institution","year","title","theme"])
        if catalogue.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(metadata)

    print(f"✅ Ajouté : {metadata['title']}")
