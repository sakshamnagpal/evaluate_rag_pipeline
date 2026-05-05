import os
import csv
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "data/chroma_db"
EVAL_CSV = "data/eval_set/questions_fixed.csv"
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_eval_set():
    questions = []
    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    return questions

def retrieve(collection, model, query, top_k=5):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0], results["metadatas"][0]

def source_match(retrieved_metadatas, expected_source_doc):
    """
    Precision metric: did we retrieve at least one chunk
    from the correct source document?
    """
    retrieved_sources = [m["source"] for m in retrieved_metadatas]
    return expected_source_doc in retrieved_sources

def source_match_at_k(retrieved_metadatas, expected_source_doc, k):
    """
    Did the correct source appear in the top-k results?
    """
    retrieved_sources = [m["source"] for m in retrieved_metadatas[:k]]
    return expected_source_doc in retrieved_sources

def evaluate(collection_name, top_k=5):
    print(f"\nEvaluating collection: {collection_name} | top_k={top_k}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(collection_name)

    questions = load_eval_set()
    results = []

    for i, q in enumerate(questions):
        docs, metadatas = retrieve(collection, model, q["question"], top_k=top_k)

        match = source_match(metadatas, q["source_doc"])
        match_at_1 = source_match_at_k(metadatas, q["source_doc"], k=1)
        match_at_3 = source_match_at_k(metadatas, q["source_doc"], k=3)

        results.append({
            "collection": collection_name,
            "source_doc": q["source_doc"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "expected_answer": q["answer"],
            "match_at_1": match_at_1,
            "match_at_3": match_at_3,
            "match_at_5": match,
            "top_retrieved_source": metadatas[0]["source"]
        })

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(questions)} questions evaluated")

    # Compute summary metrics
    total = len(results)
    precision_at_1 = sum(r["match_at_1"] for r in results) / total
    precision_at_3 = sum(r["match_at_3"] for r in results) / total
    precision_at_5 = sum(r["match_at_5"] for r in results) / total

    # Break down by difficulty
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        p_at_5 = sum(r["match_at_5"] for r in subset) / len(subset)
        print(f"  Precision@5 ({diff}): {p_at_5:.2%}")

    print(f"\n  Precision@1: {precision_at_1:.2%}")
    print(f"  Precision@3: {precision_at_3:.2%}")
    print(f"  Precision@5: {precision_at_5:.2%}")

    # Write detailed results to CSV
    output_file = os.path.join(RESULTS_DIR, f"{collection_name}_k{top_k}_results.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Detailed results saved to {output_file}")
    return precision_at_1, precision_at_3, precision_at_5

if __name__ == "__main__":
    evaluate("atsb_cs500_co100", top_k=5)