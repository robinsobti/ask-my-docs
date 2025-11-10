# Dental PDF Corpus

This repository now ships with a tiny, fixed corpus of dental reference books that live
under `data/dental_corpus/`. The crawler/normalization pipeline has been removed – these
static PDFs are the only source of truth during ingestion and retrieval.

## Included files

| Filename | Notes |
| --- | --- |
| `Dental-Anatomy-Book1.pdf` | Foundational anatomy, pulp, enamel coverage |
| `Oral_and_Maxillofacial_Surgery_E-Book.pdf` | Surgical techniques and chairside guidance |
| `Oxford Handbook of Clinical Dentistry 6e - Laura Mitchell.pdf` | Pocket reference for common clinical protocols |

The ingestion script (`scripts/ingest.py`) walks this directory, extracts text from every
PDF page with `PyPDF2`, drops front-matter pages, chunks the remaining text, and stores
embeddings together with the `book_name` metadata so each chunk remains traceable to its
parent book.

If you need to experiment with different content, drop replacement PDFs in this folder
and re-run `python scripts/ingest.py`. The filenames automatically become the `book_name`
metadata stored in the vector store.
