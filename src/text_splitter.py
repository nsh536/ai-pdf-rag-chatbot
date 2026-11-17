
from pdf_loader import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(text):

    if not text or not text.strip():
        raise ValueError("No text available for chunking.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    return chunks


def main():

    pdf_path = "data/sample.pdf"

    text = extract_text_from_pdf(pdf_path)

    chunks = split_text_into_chunks(text)

    print("\n========== TEXT CHUNKING RESULTS ==========")

    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n--- CHUNK {index} ---")
        print(chunk)

    print("\n========== CHUNKING COMPLETED ==========")


if __name__ == "__main__":
    main()
