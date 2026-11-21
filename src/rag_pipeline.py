
import os
import time
import random

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from .vector_store import load_vector_store, search_similar_chunks


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Gemini API key not found! Check your .env file."
    )

client = genai.Client(api_key=api_key)


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 3. LOAD FAISS DATABASE
# ==========================================

print("Loading FAISS vector database...")

index, chunks = load_vector_store()

print("RAG system is ready!")


# ==========================================
# 4. RETRY CONFIGURATION
# ==========================================

MAX_RETRIES = 5

RETRYABLE_ERRORS = [
    "429",
    "500",
    "502",
    "503",
    "504",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "DEADLINE_EXCEEDED"
]


def is_retryable_error(error):

    error_message = str(error).upper()

    return any(
        error_code in error_message
        for error_code in RETRYABLE_ERRORS
    )


# ==========================================
# 5. GEMINI RESPONSE WITH RETRY
# ==========================================

def generate_gemini_response(prompt):

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Sending request to Gemini "
                f"(Attempt {attempt + 1}/{MAX_RETRIES})"
            )

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )

            answer = response.text

            if not answer or not answer.strip():
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return answer.strip()

        except Exception as error:

            if not is_retryable_error(error):
                raise

            if attempt == MAX_RETRIES - 1:

                print(
                    "Gemini service unavailable after "
                    "maximum retry attempts."
                )

                return (
                    "Gemini is temporarily experiencing "
                    "high demand. Your question was not "
                    "processed. Please try again in a "
                    "little while."
                )

            # Exponential backoff with random jitter
            wait_time = min(
                2 ** (attempt + 1) + random.uniform(0, 1),
                30
            )

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_time:.1f} seconds..."
            )

            time.sleep(wait_time)


# ==========================================
# 6. MAIN RAG PIPELINE
# ==========================================

def generate_answer(question):

    question = question.strip()

    if not question:
        return "Please enter a valid question."

    # Step 1: Retrieve relevant chunks

    results = search_similar_chunks(
        question,
        embedding_model,
        index,
        chunks,
        top_k=3
    )

    if not results:
        return (
            "The answer is not available "
            "in the uploaded PDF."
        )

    # Step 2: Prepare context

    context = "\n\n".join(
        result["chunk"] for result in results
    )

    # Step 3: Create prompt

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer is not available in the context,
clearly say:

"The answer is not available in the uploaded PDF."

Do not invent information.
Do not use external knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    # Step 4: Generate Gemini response

    answer = generate_gemini_response(prompt)

    return answer


# ==========================================
# 7. TERMINAL CHAT INTERFACE
# ==========================================

def main():

    print("\n===== DOCUMIND AI =====")

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

            print("\nDocuMind AI:", answer)

        except Exception as error:

            print("\nAn unexpected error occurred:", error)


if __name__ == "__main__":
    main()
