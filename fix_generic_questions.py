import os
import csv
import fitz
import anthropic
import json
import time
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

EVAL_CSV = "data/eval_set/questions.csv"
PDF_DIR = "data/raw_pdfs"
OUTPUT_CSV = "data/eval_set/questions_fixed.csv"

GENERIC_PHRASES = [
    "what was the aircraft registration",
    "what was the registration number",
    "how many passengers",
    "what was the date of",
    "what was the wind",
    "what was the time",
    "what type of aircraft",
    "what was the flight number",
]

def is_generic(question):
    q_lower = question.lower()
    return any(phrase in q_lower for phrase in GENERIC_PHRASES)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join(page.get_text() for page in doc)

def regenerate_question(original_question, answer, difficulty, source_section, doc_text):
    prompt = f"""You are improving an evaluation dataset for a RAG system trained on aviation safety reports.

The following question is too generic and it could apply to many different reports and lacks document-specific context, making it impossible for a retrieval system to identify the correct source document.

Original question: {original_question}
Known answer: {answer}
Difficulty: {difficulty}
Source section: {source_section}

Using the report text below, rewrite the question so that it:
- Contains enough specific context that it could only refer to this particular report
- Still has the same difficulty level and tests the same fact
- Is naturally phrased as a question a real user might ask
- Does NOT include the answer within the question itself

Return only a JSON object with two fields:
- question: the rewritten question string
- reason: one sentence explaining what specific detail you added to anchor it to this document

No preamble, no markdown backticks.

Report text:
{doc_text[:6000]}
"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())

def main():
    questions = []
    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    generic_count = sum(1 for q in questions if is_generic(q["question"]))
    print(f"Found {generic_count} potentially generic questions out of {len(questions)}")

    # Cache extracted text to avoid re-reading the same PDF multiple times
    text_cache = {}

    fixed = 0
    for i, q in enumerate(questions):
        if not is_generic(q["question"]):
            continue

        source_doc = q["source_doc"]
        pdf_path = os.path.join(PDF_DIR, source_doc)

        if not os.path.exists(pdf_path):
            print(f"  PDF not found for {source_doc} — skipping")
            continue

        if source_doc not in text_cache:
            text_cache[source_doc] = extract_text(pdf_path)

        doc_text = text_cache[source_doc]

        print(f"  [{i+1}/{len(questions)}] Regenerating: {q['question'][:60]}...")

        try:
            result = regenerate_question(
                original_question=q["question"],
                answer=q["answer"],
                difficulty=q["difficulty"],
                source_section=q["source_section"],
                doc_text=doc_text
            )

            print(f"    → {result['question'][:80]}")
            print(f"    Reason: {result['reason']}")

            q["question"] = result["question"]
            q["regenerated"] = True
            fixed += 1

        except Exception as e:
            print(f"    Failed: {e}")
            q["regenerated"] = False

        time.sleep(1)

    # Write updated CSV
    fieldnames = list(questions[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions)

    print(f"\nDone. {fixed} questions regenerated.")
    print(f"Updated eval set saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()