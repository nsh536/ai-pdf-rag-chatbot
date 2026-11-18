
import os
import time

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from .vector_store import load_vector_store, search_similar_chunks


# Load environment variables
load_dotenv()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

client = genai.Client(api_key=api_key)


# Load embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load existing FAISS database
print("Loading FAISS vector database...")
index, chunks = load_vector_store()

print("RAG system is ready!")


def generate_answer(question):

    # Step 1: Retrieve relevant chunks
    results = search_similar_chunks(
        question,
        embedding_model,
        index,
        chunks,
        top_k=3
    )

    if not results:
        return "I could not find relevant information in the PDF."

    # Step 2: Prepare context
    context = "\n\n".join(
        result["chunk"] for result in results
    )

    # Step 3: Create prompt
    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the information provided
in the context below.

If the answer is not available in the context,
clearly say:
"The answer is not available in the uploaded PDF."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""

    # Step 4: Generate answer with automatic retry
    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            return response.text

        except Exception as error:

            error_message = str(error)

            if "503" in error_message and attempt < 2:

                print(
                    f"Gemini busy. Retrying... ({attempt + 1}/3)"
                )

                time.sleep(3)

            else:
                raise


def main():

    print("\n===== AI PDF RAG CHATBOT =====")
    print("Ask questions about your PDF.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Chat session ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = generate_answer(question)

            print("\nGemini:", answer)

        except Exception as error:
            print("\nError:", error)


if __name__ == "__main__":
    main()
