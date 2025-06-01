**RAG PDF Q\&A Application**
*A Retrieval-Augmented Generation (RAG) demo that lets users upload a PDF, embed its contents, and ask natural-language questions—powered by Chroma, HuggingFace embeddings, and Gemini.*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture & Workflow](#architecture--workflow)
4. [Directory Structure](#directory-structure)
5. [Setup & Installation](#setup--installation)
6. [Usage](#usage)
7. [Key Components](#key-components)
8. [Dependencies](#dependencies)
9. [Future Improvements](#future-improvements)
10. [License](#license)

---

## Project Overview

This repository contains a minimal-but-robust example of a **RAG (Retrieval-Augmented Generation)** pipeline that:

1. **Ingests** a PDF file.
2. **Splits** its text into overlapping chunks.
3. **Embeds** each chunk into a vector database (Chroma) using a lightweight HuggingFace model.
4. **Retrieves** the top-K most relevant chunks for any user query.
5. **Generates** a natural-language answer by sending the retrieved context + question to Google’s Gemini LLM.
6. **Exposes** everything via a Streamlit “dashboard” so end-users (or recruiters/technical reviewers) can upload a PDF, wait a few seconds for embeddings, then ask questions in plain English.

By the end of this project, you can demonstrate:

* Familiarity with **vector-search** and **embedding pipelines**.
* Integration of a **Chroma** vector store (persistent on‐disk DB).
* Usage of **HuggingFace embeddings** (“all-MiniLM-L6-v2”) in Python.
* Prompt‐engineering style: combining “retrieved context” + “user question.”
* Calling a modern LLM (“gemini-2.0-flash”) via the `google-generativeai` SDK.
* A basic but practical **Streamlit** frontend for PDF upload + Q\&A.

This kind of repo is exactly what you’d showcase to recruiters to highlight your end-to-end AI/ML + software‐engineering chops: data‐ingestion, LLM integration, vector DBs, and a user-friendly interface.

---

## Features

* **PDF Upload**: Users can drag & drop or browse to upload any PDF.
* **Chunking & Embeddings**: Automatically splits text into \~1,000-char overlapping windows, computes embeddings with a pre-trained Sentence-Transformer, and persists them in Chroma.
* **Vector Search**: For every free-text query, fetches the top-K matching chunks from Chroma.
* **LLM Generation**: Feeds the retrieved context + question into Gemini-2.0-Flash, returning a concise, conversational answer aimed at non-technical audiences.
* **Streamlit UI**: A single‐page dashboard where you can “Process PDF” (builds a temp DB) and then “Ask” questions interactively.
* **Modular Code**: The core RAG logic (`rag.py`) is decoupled from UI, making it easy to swap in different embeddings or LLMs later.
* **CLI Fallback**: If you run `python rag.py` directly, you get a simple REPL (Ctrl+C safe) to ask questions in the terminal.

---

## Architecture & Workflow

1. **PDF Ingestion**

   * User uploads a PDF in the Streamlit app.
   * The app saves it to `./data/<filename>.pdf`.

2. **Text Splitting & Embedding**

   * `generative_embeddings.py` (invoked by Streamlit on‐the-fly) loads the PDF via `PyPDFLoader`, splits it using `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)`, and collects \~1,000-char chunks (with 100-char overlap).
   * Each chunk is embedded using

     ```python
     HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
     ```
   * Chroma writes these vectors to `./temp_db` (Streamlit session) or `./chroma_db_nccn` (if you ran `python generative_embeddings.py` manually before).

3. **Vector Retrieval**

   * In `rag.py`, `get_db_context(query, persist_directory)` loads the Chroma DB and conducts a similarity search (top-K by default).
   * Returns the concatenated “page\_content” of the top matches as a single string.

4. **Prompt Construction & LLM Call**

   * `generate_rag_prompt(query, context)` builds a template:

     ```
     You are a helpful, non-technical assistant. Use ONLY the reference CONTEXT below to answer the QUESTION...
     QUESTION:
     <user question>

     CONTEXT:
     <concatenated chunks>

     ANSWER:
     ```
   * `generate_answer(prompt)` calls `google.generativeai.GenerativeModel("gemini-2.0-flash")` with your `GEMINI_API_KEY`.
   * Returns `response.text` (the generated answer).

5. **Streamlit Frontend**

   * Renders a two-step interface:

     1. **Process PDF** button
     2. **Ask** button + text input
   * Persists the embedding directory in `st.session_state["db_dir"]`.
   * Once embeddings exist, the “Ask” action retrieves context, builds a prompt, calls Gemini, and displays the result.

6. **CLI Mode (Optional)**

   * If you do `python rag.py`, it registers a Ctrl+C handler, then enters a loop:

     1. Ask “What question do you have about the PDF?”
     2. Fetch context via Chroma (default `./chroma_db_nccn`)
     3. Generate answer via Gemini
     4. Print it in the terminal
   * Stops gracefully on Ctrl+C.

---

## Directory Structure

```
.
├── README.md
├── data
│   └── (Place your own PDF files here, or Streamlit will create them)
├── chroma_db_nccn        # (Auto-created by generative_embeddings.py)
│   └── (persistent Chroma vector store for ‘official’ corpus)
├── temp_db               # (Created/replaced each Streamlit run)
│   └── (temporary Chroma DB for the PDF you just uploaded)
├── generative_embeddings.py
├── rag.py
├── streamlit_app.py
└── requirements.txt      # (Optional: list of pip dependencies)
```

* **`data/`**

  * Where your PDF files live.
  * Streamlit saves the uploaded PDF here (e.g. `data/mydoc.pdf`).
  * `generative_embeddings.py` also expects PDFs under `data/`.

* **`chroma_db_nccn/`**

  * If you want a long-term, re-usable vector store for a static set of PDFs, run:

    ```bash
    python generative_embeddings.py
    ```
  * That script will scan `data/gpu.pdf` (you can modify to load multiple), split into chunks, embed, and write \~52 vectors.
  * Later, if you run `python rag.py` in CLI mode, it will default to searching this directory.

* **`temp_db/`**

  * Each time you hit “Process PDF” in Streamlit, it generates a new Chroma DB here (overwriting previous).
  * Subsequent “Ask” calls read from this folder so each session is isolated.

* **`generative_embeddings.py`**

  * One-off script to load PDF(s), split text, embed, and write to `./chroma_db_nccn`.
  * Handy if you want to pre-build embeddings for a “known” document set.

* **`rag.py`**

  * Core RAG logic:

    * `generate_rag_prompt(...)`
    * `get_db_context(...)` (vector retrieval)
    * `generate_answer(...)` (LLM call)
  * Also includes a CLI entrypoint if run as `__main__`.

* **`streamlit_app.py`**

  * Single-page web UI to upload, embed, and ask questions.
  * Reads/writes to `./data` and `./temp_db`.
  * Calls into `rag.py` for retrieval + generation.

---

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/pdf-rag-project.git
   cd pdf-rag-project
   ```

2. **Create a virtual environment** (highly recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   * Create a file `requirements.txt` containing:

     ```
     streamlit
     langchain_community
     chromadb
     sentence-transformers
     google-generativeai
     tf-keras          # required shim for Transformers <→ Keras 3 compatibility
     ```
   * Then run:

     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

   **Alternatively**, install manually:

   ```bash
   pip install streamlit langchain_community chromadb sentence-transformers google-generativeai tf-keras
   ```

4. **Set environment variables**

   * **Gemini API Key** (replace with your real key):

     ```bash
     export GEMINI_API_KEY="AIzaSyDEQrNa3PI3yFY_PrRm3hXadoB3KesnboA"
     ```
   * On Windows (PowerShell):

     ```powershell
     setx GEMINI_API_KEY "AIzaSy..." 
     ```
   * *Optional:* If you switch to `OpenAIEmbeddings` instead of HuggingFace, set:

     ```bash
     export OPENAI_API_KEY="<your-openai-key>"
     ```

5. **(Optional) Prebuild your “official” Chroma DB**
   If you want a permanent, multi-PDF vector store under `./chroma_db_nccn`, open `generative_embeddings.py`, point it at any PDFs you like in `data/`, then:

   ```bash
   python generative_embeddings.py
   # → prints “Persisted X vectors.”
   ```

   From now on, `python rag.py` in CLI mode will search `./chroma_db_nccn`.

6. **Run the Streamlit app**

   ```bash
   streamlit run streamlit_app.py
   ```

   * The browser will open at `http://localhost:8501`.
   * Step 1: click **Process PDF**, choose your PDF file.
   * Step 2: once “✅ Embeddings built!” appears, type a question and click **Ask**.
   * You’ll see Gemini’s answer in a friendly, non-technical style.

---

## Usage

### CLI Mode (Optional)

If you prefer a simple terminal experience (no Streamlit), you can interact directly with `rag.py`:

1. Make sure you have already run `generative_embeddings.py` to populate `./chroma_db_nccn`.
2. In your terminal:

   ```bash
   python rag.py
   ```
3. The script will ask:

   ```
   What question do you have about the PDF? 
   ```
4. Type any free-text query (e.g. “Explain GPU architecture”).
5. Hit Enter. You’ll see a Gemini-generated answer printed.
6. Press `Ctrl+C` to exit gracefully (it registers a handler that prints “thanks for using rag app with gemini”).

### Streamlit UI

1. **Upload a PDF**

   * Click “Browse files” or drag your PDF into the “Upload a PDF” widget.
   * Click **Process PDF**.
   * You’ll see “Making embeddings…” followed by “✅ Embeddings built!” once complete (usually <10 seconds for a 2-page PDF).

2. **Ask Questions**

   * A text box appears labeled “Ask me anything about your upload:”.
   * Type your question (e.g. “Summarize Section 2.3 in simple terms”).
   * Click **Ask**.
   * Under “Answer:”, you’ll get Gemini’s response, which references the retrieved context and is phrased for non-technical audiences.

3. **Reprocess / New PDF**

   * To switch PDFs, simply re-upload and click **Process PDF** again. The old `./temp_db` is overwritten.

---

## Key Components

### 1. **`generative_embeddings.py`**

* **Purpose**: One‐time embedding build for a “static” corpus.
* **Workflow**:

  1. Load PDF(s) from `data/` via `PyPDFLoader`.
  2. Split pages into overlapping chunks (1,000 characters ±100 overlap).
  3. Embed each chunk with `sentence-transformers/all-MiniLM-L6-v2`.
  4. Persist vectors into `./chroma_db_nccn` via Chroma.

> **Tip**: If you want multiple PDFs in the “official” store, update the `loaders` list to include all filepaths under `data/`, rerun, and you’ll get a combined store.

### 2. **`rag.py`**

* **Purpose**: Core retrieval + generation logic, split into three re-usable functions + optional CLI.

  1. `generate_rag_prompt(query: str, context: str) -> str`:

     * Cleans quotes/newlines from `context`.
     * Fills a prompt template reminding the LLM to “answer only from the context” in plain English.
  2. `get_db_context(query: str, k: int = 6, persist_directory: str = "./chroma_db_nccn") -> str`:

     * Loads a Chroma DB at `persist_directory`.
     * Uses the same `HuggingFaceEmbeddings(…, device='cpu')` to encode `query`.
     * Runs `similarity_search(query, k)` and returns the concatenated `page_content` of the top-K docs.
  3. `generate_answer(prompt: str) -> str`:

     * Configures `genai` with `GEMINI_API_KEY`.
     * Calls `GenerativeModel("gemini-2.0-flash")`.
     * Returns `response.text`.

* **CLI Entrypoint** (only if you run `python rag.py` directly):

  * Registers a Ctrl+C handler that prints “thanks for using rag app with gemini” and exits.
  * Loops: ask for user query → fetch context → build prompt → call Gemini → print answer.

### 3. **`streamlit_app.py`**

* **Purpose**: Single-page web application that ties everything together.
* **Key Steps**:

  1. **PDF upload widget** (`st.file_uploader(type=["pdf"])`).
  2. **“Process PDF” button**:

     * Saves the upload to `data/<filename>`.
     * Imports `PyPDFLoader`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, and `Chroma` from `generative_embeddings.py`.
     * Splits & embeds into `./temp_db`.
     * Sets `st.session_state["db_dir"] = "./temp_db"`.
  3. **“Ask” button + text\_input**:

     * If `db_dir` exists, user can type a question.
     * On click: call `get_db_context(query, persist_directory=st.session_state["db_dir"])`.
     * Build prompt & call `generate_answer(prompt)`.
     * Display the returned answer.

---

## Dependencies

Below is a non-exhaustive list of the core pip packages required. Your `requirements.txt` might look like:

```
streamlit                # for the dashboard
langchain_community      # embedding and Chroma helpers
chromadb                 # persistent vector DB
sentence-transformers    # miniLM embeddings
google-generativeai      # Gemini LLM client
tf-keras                 # compatibility layer for Transformers <→ Keras 3
```

You can install all at once:

```bash
pip install streamlit langchain_community chromadb sentence-transformers google-generativeai tf-keras
```

*(If you swap to `OpenAIEmbeddings` instead of `HuggingFaceEmbeddings`, add `openai` and remove `tf-keras`.)*

---

## Future Improvements

1. **Multi-PDF Support**

   * Right now, Streamlit always overwrites `./temp_db`. You could enable uploading multiple PDFs in a single session and merging their embeddings before answering.

2. **Improved Chunking**

   * Use smarter splitters (e.g. `TokenTextSplitter`) that respect sentence boundaries or custom heuristics.

3. **Caching & Incremental Updates**

   * Instead of fully rebuilding embeddings each time, detect if an identical PDF already exists (hash by filename+size) and reuse its `temp_db`.
   * For large corpora, implement incremental ingestion (only new pages or docs).

4. **Alternative Embeddings**

   * Swap in any other HuggingFace SentenceTransformer, SBERT, or even OpenAI embeddings for potentially better retrieval quality (given an OpenAI API key).

5. **Prompt-Aggregation & Reranking**

   * Implement a reranker (e.g. cross-encoder) to refine the top-K shortlist before passing to Gemini.
   * Add chain-of-thought or “answer verification” steps if you want higher accuracy.

6. **Localization & UI Polishing**

   * Add progress bars for chunking/embedding steps.
   * Display a snippet preview of the retrieved chunks.
   * Style the Streamlit page (custom CSS, layout improvements, dark/light mode).

7. **Deployment**

   * Containerize with Docker and deploy on Heroku/Render/Vercel.
   * Automate secret management (store `GEMINI_API_KEY` in a GitHub Secrets vault or environment variable).

---

## License

This project is released under the [MIT License](LICENSE). Feel free to copy, modify, and redistribute as you see fit, provided you keep the original attribution.

---

**Congratulations!** You now have a fully functional RAG system that demonstrates:

* Handling PDF-to-vector workflows
* Vector similarity search
* LLM integration (Gemini)
* A user-friendly React-style UI (Streamlit)

Feel free to point any recruiter or collaborator to this repo to showcase your ability to build end-to-end AI applications in Python.
