"""ingest_folder.py — Ingère les PDFs décrits dans un catalogue CSV."""
import argparse
import csv
from pathlib import Path
from rag_public_reports.config import DATA_DIR
from rag_public_reports.ingest import ingest_pdf
from rag_public_reports.vectorstore import get_vector_store, add_documents, is_already_ingested

parser = argparse.ArgumentParser(description="Ingère les PDFs d'un catalogue CSV")
parser.add_argument("--folder", type=str, default=str(DATA_DIR / "raw"))
args = parser.parse_args()

folder = Path(args.folder)
catalogue = folder / "catalogue.csv"

if not catalogue.exists():
    print(f"❌ Fichier catalogue.csv introuvable dans {folder}")
    exit(1)

vs = get_vector_store()

with open(catalogue, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        pdf = folder / row["fichier"]
        if not pdf.exists():
            print(f"⚠️  PDF introuvable : {pdf.name} — ignoré")
            continue
        if is_already_ingested(vs, str(pdf)):
            continue
        chunks = ingest_pdf(pdf, row["institution"], int(row["year"]), row["title"], row["theme"])
        add_documents(vs, chunks)
