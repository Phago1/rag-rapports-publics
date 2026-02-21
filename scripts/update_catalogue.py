"""update_catalogue.py — Extrait les métadonnées des PDFs et met à jour catalogue.csv."""
import sys
from pathlib import Path

sys.path.insert(0, "src")

from rag_public_reports.catalogue import extraire_metadata, ajouter_au_catalogue
from rag_public_reports.config import DATA_DIR

PDF_DIR = DATA_DIR / "raw"
CATALOGUE_PATH = PDF_DIR / "catalogue.csv"

pdfs = list(PDF_DIR.glob("*.pdf"))
print(f"📂 {len(pdfs)} PDFs trouvés dans {PDF_DIR}")

for pdf in pdfs:
    print(f"\n🔍 Traitement : {pdf.name}")
    try:
        metadata = extraire_metadata(str(pdf))
        ajouter_au_catalogue(metadata, catalogue_path=str(CATALOGUE_PATH))
    except Exception as e:
        print(f"❌ Erreur sur {pdf.name} : {e}")
