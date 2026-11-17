
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_loader import extract_text_from_pdf
from text_splitter import split_text_into_chunks


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "sample.pdf"
STORAGE_DIR = BASE_DIR / "storage"

INDEX_PATH = STORAGE_DIR / "document.index"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def create_vector_store(model):

    print("\nReading PDF...")

    # Extract PDF text
    text = extract_text_from_pdf(str(PDF_PATH))

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    if not chunks:
        raise ValueError("No text chunks found in PDF.")

    print("\nGenerating embeddings...")

    # Convert chunks into normalized vectors
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Store embeddings
    index.add(embeddings)

    # Create storage folder
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(INDEX_PATH))

    # Save original text chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

    print("\nFAISS vector database created successfully!")
    print("Total vectors stored:", index.ntotal)
    print("Vector dimensions:", dimension)

    return index, chunks


def load_vector_store():

    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Vector database not found. Create it first."
        )

    # Load saved FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # Load corresponding text chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunks do not match.")

    return index, chunks


def search_similar_chunks(question, model, index, chunks, top_k=3):

    if not question.strip():
        raise ValueError("Please enter a question.")

    # Convert question into an embedding
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.ascontiguousarray(
        question_embedding,
        dtype=np.float32
    )

    # Search similar vectors
    scores, indices = index.search(
        question_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "chunk": chunks[int(idx)],
            "similarity_score": float(score),
            "chunk_index": int(idx)
        })

    return results


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # Create a fresh index from our PDF
    index, chunks = create_vector_store(model)

    print("\n========== SEMANTIC SEARCH ==========")
    print("Ask questions about your PDF.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nEnter your question: ")

        if question.lower().strip() == "exit":
            print("Search session ended.")
            break

        try:
            results = search_similar_chunks(
                question,
                model,
                index,
                chunks,
                top_k=3
            )

            print("\n========== SEARCH RESULTS ==========")

            for rank, result in enumerate(results, start=1):

                print(f"\nResult {rank}")
                print("Chunk index:", result["chunk_index"])
                print(
                    "Similarity score:",
                    round(result["similarity_score"], 4)
                )
                print("Content:", result["chunk"])

        except ValueError as error:
            print("Error:", error)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nApplication error: {error}")
