
from pypdf import PdfReader
from pathlib import Path


def extract_text_from_pdf(pdf_path):

    # Check whether PDF exists
    file_path = Path(pdf_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    # Check PDF extension
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    # Read PDF
    reader = PdfReader(str(file_path))

    extracted_pages = []

    print(f"Total Pages: {len(reader.pages)}")

    # Extract text page by page
    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        extracted_pages.append(text)

        print(f"Page {page_number} extracted successfully.")

    # Combine all page text
    full_text = "\n".join(extracted_pages)

    return full_text


if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    try:
        text = extract_text_from_pdf(pdf_path)

        print("\n========== EXTRACTED PDF TEXT ==========\n")
        print(text)

        print("\n========== EXTRACTION COMPLETED ==========")

    except Exception as error:
        print(f"Error: {error}")
