import streamlit as st
import os
from dotenv import load_dotenv

# ---------------- LOAD API KEY ---------------- #

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ---------------- IMPORTS ---------------- #

from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

# ---------------- GROQ CLIENT ---------------- #

client = Groq(api_key=api_key)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AskMyPDF AI",
    layout="wide"
)

st.title("AskMyPDF AI")

# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome! Upload a PDF and start chatting."
        }
    ]

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "embed_model" not in st.session_state:
    st.session_state.embed_model = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# 🔥 Important for resetting file uploader
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.header("⚙️ Controls")

    # -------- CLEAR CHAT + PDF -------- #

    if st.button("🗑 Clear Chat"):

        # Clear messages
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome! Upload a PDF and start chatting."
            }
        ]

        # Clear vector DB
        st.session_state.index = None
        st.session_state.chunks = None
        st.session_state.embed_model = None

        # Remove uploaded PDF tracking
        st.session_state.current_file = None

        # Reset uploader completely
        st.session_state.uploader_key += 1

        st.toast("🗑 Chat and PDF cleared!", icon="✅")

        st.rerun()

    st.markdown("### Example Questions")

    st.write("- Summarize this document")
    st.write("- What are key points?")
    st.write("- Explain the main topic")
    st.write("- Give important insights")

# ---------------- FILE UPLOADER ---------------- #

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf",
    key=f"uploader_{st.session_state.uploader_key}"
)

# ---------------- HANDLE NEW PDF ---------------- #

if uploaded_file:

    # Detect NEW uploaded file
    if st.session_state.current_file != uploaded_file.name:

        st.session_state.current_file = uploaded_file.name

        # Reset old chat
        st.session_state.messages = []

        # Reset vector DB
        st.session_state.index = None
        st.session_state.chunks = None
        st.session_state.embed_model = None

        st.toast(
            "📄 New document uploaded! Chat reset.",
            icon="📄"
        )

# ---------------- PROCESS PDF ---------------- #

if uploaded_file and st.session_state.index is None:

    with st.spinner("Processing PDF..."):

        # Read PDF
        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text

        # -------- CHUNKING -------- #

        chunk_size = 500

        chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

        # -------- EMBEDDINGS -------- #

        embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        embeddings = embed_model.encode(chunks)

        # -------- FAISS VECTOR DB -------- #

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(np.array(embeddings))

        # -------- STORE -------- #

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.embed_model = embed_model

    st.success("✅ PDF processed successfully!")

# ---------------- DISPLAY CHAT ---------------- #

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

# ---------------- CHATBOT ---------------- #

if st.session_state.index is not None:

    query = st.chat_input(
        "Ask a question about the PDF..."
    )

    if query:

        # -------- USER MESSAGE -------- #

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # -------- QUERY EMBEDDING -------- #

        query_vector = st.session_state.embed_model.encode([query])

        # -------- SEARCH RELEVANT CHUNKS -------- #

        distances, indices = st.session_state.index.search(
            query_vector,
            3
        )

        # -------- BUILD CONTEXT -------- #

        context = "\n".join(
            [
                st.session_state.chunks[i]
                for i in indices[0]
            ]
        )

        # -------- PROMPT -------- #

        prompt = f"""
        You are an AI assistant.

        Answer ONLY from the given context.

        If the answer is not clearly present in the context,
        strictly reply with:

        "This question is not related to the uploaded document."

        Context:
        {context}

        Question:
        {query}
        """

        # -------- GENERATE RESPONSE -------- #

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        # -------- SAVE RESPONSE -------- #

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # 🔥 Force instant refresh
        st.rerun()

# ---------------- EMPTY STATE ---------------- #

else:

    st.info(
        "Please upload a PDF to start chatting."
    )