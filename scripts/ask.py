"""ask.py — Pose une question au RAG depuis le terminal."""
import argparse
from rag_public_reports.rag import answer

parser = argparse.ArgumentParser(description="Interroge les rapports indexés")
parser.add_argument("question", type=str, help="Ta question en langage naturel")
parser.add_argument("--institution", type=str, default=None)
parser.add_argument("--year", type=int, default=None)
parser.add_argument("--theme", type=str, default=None)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

print(answer(
    args.question,
    filter_institution=args.institution,
    filter_year=args.year,
    filter_theme=args.theme,
    verbose=args.verbose,
))
