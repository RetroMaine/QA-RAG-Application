# streamlit_app.py
import os
import streamlit as st
from rag import get_db_context, generate_rag_prompt, generate_answer

st.set_page_config(page_title="📚 RAG Dashboard", layout="wide")
st.title("📚 RAG Q&A Dashboard")

# 1) PDF upload & embedding
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if st.button("Process PDF") and uploaded:
    # 1a) Save
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # 1b) Embed into a temp folder
    st.spinner("🛠 Building embeddings…")
    from generative_embeddings import PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, Chroma

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    embed_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    temp_dir = "./temp_db"
    # embedding_function=embed_fn
    Chroma.from_documents(docs, embed_fn, persist_directory=temp_dir)
    st.success("✅ Embeddings built!")
    st.session_state["db_dir"] = temp_dir

# 2) Ask questions
if "db_dir" in st.session_state:
    query = st.text_input("Ask me anything about your upload:")
    if st.button("Ask") and query.strip():
        with st.spinner("🤖 Retrieving & generating…"):
            ctx = get_db_context(query, persist_directory=st.session_state["db_dir"])
            prompt = generate_rag_prompt(query, ctx)
            answer = generate_answer(prompt)
        st.markdown("**Answer:**")
        st.write(answer)
