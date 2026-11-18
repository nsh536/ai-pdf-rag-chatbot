
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Send a test prompt
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain Artificial Intelligence in two simple sentences."
)

print("\nGemini Response:")
print(response.text)
