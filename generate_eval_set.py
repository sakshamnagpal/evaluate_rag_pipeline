import os
import csv
import fitz  # pymupdf
import anthropic
from dotenv import load_dotenv
import json
import time

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PDF_DIR = "data/raw_pdfs"
OUTPUT_CSV = "data/eval_set/questions.csv"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join(page.get_text() for page in doc)

def generate_questions(text, source_doc):
    prompt = f"""You are building an evaluation dataset for a RAG (Retrieval-Augmented Generation) system trained on aviation safety investigation reports.

Given the following report text, generate exactly 5 question-answer pairs of varying difficulty:
- 2 easy questions (direct lookup of a specific fact)
- 2 medium questions (require understanding of a specific section)
- 1 hard question (requires synthesising information across multiple sections)

Return your response as a JSON array with exactly 5 objects, each with these fields:
- question: the question string
- answer: the expected answer string
- difficulty: one of "easy", "medium", "hard"
- source_section: the section of the report the answer comes from

Return only the JSON array. No preamble, no explanation, no markdown backticks.

Report text:
{text[:8000]}
"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]  # take content between fences
        if raw.startswith("json"):
            raw = raw[4:]  # strip the "json" language tag

    return raw.strip()

def main():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["source_doc", "difficulty", "question", "answer", "source_section"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            print(f"[{i+1}/{len(pdf_files)}] Processing {pdf_file}...")

            try:
                text = extract_text(pdf_path)

                if len(text.strip()) < 500:
                    print(f"  Skipping — insufficient text extracted (possibly scanned)")
                    continue

                raw = generate_questions(text, pdf_file)
                questions = json.loads(raw)

                for q in questions:
                    writer.writerow({
                        "source_doc": pdf_file,
                        "difficulty": q["difficulty"],
                        "question": q["question"],
                        "answer": q["answer"],
                        "source_section": q["source_section"]
                    })

                print(f"  Done — {len(questions)} questions written")

                # Avoid hitting API rate limits
                time.sleep(2)

            except json.JSONDecodeError:
                print(f"  Failed to parse JSON response for {pdf_file} — skipping")
            except Exception as e:
                print(f"  Error processing {pdf_file}: {e}")

    print(f"\nComplete. Eval set saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()