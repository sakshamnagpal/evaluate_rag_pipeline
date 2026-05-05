# RAG Evaluation

A pipeline for evaluating Retrieval-Augmented Generation (RAG) systems on aviation safety investigation reports.

## Overview

This project:
1. Ingests PDF reports into a ChromaDB vector store with configurable chunking parameters
2. Uses Claude to auto-generate a labelled evaluation dataset (question-answer pairs) from the same PDFs
3. (Results stored in `data/results/` for downstream evaluation)

## Project Structure

```
rag_evaluation/
├── ingest.py              # PDF ingestion → ChromaDB
├── generate_eval_set.py   # Claude-powered eval set generation
├── data/
│   ├── raw_pdfs/          # Input PDF reports
│   ├── chroma_db/         # Persistent ChromaDB vector store
│   ├── eval_set/          # Generated questions.csv
│   └── results/           # Evaluation outputs
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install chromadb langchain-text-splitters sentence-transformers pymupdf anthropic python-dotenv
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Place PDF reports in `data/raw_pdfs/`.

## Usage

**1. Ingest PDFs into the vector store**

```bash
python ingest.py
```

This runs two ingestion passes with different chunk overlap settings (`chunk_overlap=0` and `chunk_overlap=100`), creating separate ChromaDB collections named `atsb_cs{chunk_size}_co{chunk_overlap}`.

**2. Generate the evaluation dataset**

```bash
python generate_eval_set.py
```

Generates 5 question-answer pairs per PDF (2 easy, 2 medium, 1 hard) using Claude, saved to `data/eval_set/questions.csv`.

## Dependencies

| Package | Version |
|---|---|
| chromadb | 1.5.8 |
| langchain-text-splitters | 1.1.2 |
| sentence-transformers | 5.4.1 |
| PyMuPDF | 1.27.2 |
| anthropic | 0.97.0 |
