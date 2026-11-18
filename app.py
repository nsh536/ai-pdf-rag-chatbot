
import streamlit as st
from src.rag_pipeline import generate_answer

# Page configuration
st.set_page_config(
    page_title="AI PDF RAG Chatbot",
    page_icon="📚",
    layout="centered"
)

# Application title
st.title("📚 AI PDF RAG Chatbot")
st.caption("Ask questions and get answers from your PDF using AI.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input("Ask something about your PDF...")

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            try:
                answer = generate_answer(question)
                st.markdown(answer)

            except Exception as error:
                answer = f"An error occurred: {error}"
                st.error(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot uses FAISS for semantic search "
        "and Google Gemini for AI-generated answers."
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
