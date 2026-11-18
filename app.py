
import streamlit as st
import textwrap

from src.rag_pipeline import generate_answer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. LOAD CUSTOM CSS
# ==========================================

def load_css():
    with open("assets/style.css", "r", encoding="utf-8") as file:
        css = file.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )


load_css()


# ==========================================
# 3. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        textwrap.dedent("""
        <div style="padding: 15px 5px;">
            <h2 style="color: white;">📚 DocuMind AI</h2>
            <p style="color: #aab4d0;">
                Your intelligent PDF assistant
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ✨ About")

    st.write(
        "Ask questions about your PDF and get "
        "AI-powered answers using semantic search."
    )

    st.markdown("### ⚙️ Technology")

    st.markdown("""
    - 🧠 Google Gemini AI
    - 🔍 FAISS Vector Search
    - 📄 PDF Text Extraction
    - 🔗 Sentence Transformers
    """)

    st.divider()

    st.markdown("### 💬 Chat Controls")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        textwrap.dedent("""
        <div style="
            padding-top: 25px;
            color: #8792b0;
            font-size: 12px;
        ">
            DocuMind AI · Glass Edition
        </div>
        """),
        unsafe_allow_html=True
    )


# ==========================================
# 5. MAIN HEADER
# ==========================================

st.html("""
<div style="text-align: center; padding: 25px 10px 30px;">

    <div style="
        font-size: 13px;
        color: #a78bfa;
        letter-spacing: 3px;
        font-weight: 700;
    ">
        YOUR PERSONAL AI RESEARCH ASSISTANT
    </div>

    <h1 class="hero-title">
        Chat with your
        <span class="hero-gradient">PDF Documents</span>
    </h1>

    <p class="hero-subtitle">
        Upload knowledge. Ask questions. Get intelligent answers.
    </p>

</div>
""")


# ==========================================
# 6. WELCOME CARD
# ==========================================

# WELCOME CARD

if not st.session_state.messages:

    st.html("""
    <div class="glass-card" style="text-align:center;">

        <div style="font-size:42px;">📖</div>

        <h3 style="color:white;">
            Welcome to DocuMind AI
        </h3>

        <p style="color:#aab4d0;">
            Your document is ready for intelligent search.
            Ask a question to get started.
        </p>

        <div style="
            display:flex;
            justify-content:center;
            gap:12px;
            flex-wrap:wrap;
            margin-top:18px;">

            <span style="
                background:rgba(167,139,250,0.15);
                padding:9px 15px;
                border-radius:12px;
                color:#c4b5fd;">
                🧠 AI Powered
            </span>

            <span style="
                background:rgba(96,165,250,0.15);
                padding:9px 15px;
                border-radius:12px;
                color:#93c5fd;">
                🔍 Smart Search
            </span>

            <span style="
                background:rgba(52,211,153,0.15);
                padding:9px 15px;
                border-radius:12px;
                color:#6ee7b7;">
                📄 PDF Based
            </span>

        </div>
    </div>
    """)

# ==========================================
# 7. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# 8. CHAT INPUT
# ==========================================

question = st.chat_input(
    "✨ Ask anything about your PDF..."
)


# ==========================================
# 9. GENERATE AI RESPONSE
# ==========================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🧠 Analyzing your document..."):

            try:
                answer = generate_answer(question)
                st.markdown(answer)

            except Exception as error:
                answer = f"An error occurred: {error}"
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
