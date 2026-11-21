# 📚 DocuMind AI – AI-Powered PDF RAG Chatbot

DocuMind AI is an AI-powered PDF question-answering application that allows users to ask questions about document content and receive context-based answers using Retrieval-Augmented Generation (RAG).

The application combines semantic search with Google Gemini to retrieve relevant information from an indexed PDF and generate answers based on the retrieved context.

## ✨ Features

- PDF text extraction
- Intelligent text chunking
- Semantic search using FAISS
- Text embeddings using Sentence Transformers
- AI-generated answers using Google Gemini
- Context-based question answering
- Handles questions outside the document context
- Interactive Streamlit chat interface
- Glassmorphism-inspired UI design
- Chat history and clear chat functionality

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| PDF Processing | PyPDF |
| Text Splitting | LangChain Text Splitters |
| Environment Variables | python-dotenv |

## 🏗️ Project Architecture

```text
PDF Document
     |
     v
PDF Text Extraction
     |
     v
Text Chunking
     |
     v
Text Embeddings
     |
     v
FAISS Vector Store
     |
     v
User Question
     |
     v
Semantic Search
     |
     v
Relevant Context
     |
     v
Google Gemini
     |
     v
Context-Based Answer
```

## 📂 Project Structure

```text
ai-pdf-rag-chatbot/
│
├── app.py
├── assets/
│   └── style.css
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag_pipeline.py
│
├── data/
│   └── sample.pdf
│
├── storage/
│   ├── chunks.json
│   └── document.index
│
├── .env
├── .gitignore
└── requirements.txt
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/nsh536/ai-pdf-rag-chatbot.git
cd ai-pdf-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own Gemini API key.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🧠 How It Works

1. The PDF document is processed to extract its text.
2. Extracted text is divided into smaller chunks.
3. Sentence Transformers converts chunks into numerical embeddings.
4. FAISS stores the embeddings and performs similarity search.
5. When a user asks a question, the system retrieves relevant chunks.
6. Retrieved context is passed to Google Gemini.
7. Gemini generates an answer based on the retrieved document context.

## 🔐 Environment Variables

The application requires a Gemini API key.

Never commit your `.env` file or expose API keys publicly.

## 🚀 Future Enhancements

- Upload and process multiple PDF documents
- Support for different document formats
- Document selection and management
- Chat session persistence
- Deployment to a cloud hosting platform

## 👨‍💻 Author

Developed as a hands-on AI/ML project to explore Retrieval-Augmented Generation, semantic search, vector databases, and Large Language Models.

## 📄 License

This project is available for educational and personal learning purposes.
