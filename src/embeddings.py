
from pdf_loader import extract_text_from_pdf
from text_splitter import split_text_into_chunks
from sentence_transformers import SentenceTransformer


def generate_embeddings(chunks):

    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text chunks into numerical vectors
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


def main():

    pdf_path = "data/sample.pdf"

    # Step 1: Extract PDF text
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Split text into chunks
    chunks = split_text_into_chunks(text)

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)

    print("\n========== EMBEDDING RESULTS ==========")

    print("Total chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Vector dimensions:", embeddings.shape[1])

    print("\nFirst chunk embedding:")
    print(embeddings[0][:10])

    print("\n========== EMBEDDINGS COMPLETED ==========")


if __name__ == "__main__":
    main()
