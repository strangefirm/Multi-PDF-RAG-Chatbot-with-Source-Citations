# Multi-PDF RAG Chatbot with Source Citations

A Streamlit application that answers questions across **several PDF documents at once** and shows exactly where each answer came from: the **file name and page number**, so every claim can be checked.

Built with LangChain, FAISS, Hugging Face / OpenAI embeddings and OpenAI `gpt-4o-mini`.

## Project Description

Searching through many policy documents, handbooks or reports by hand is slow, and a chatbot that answers without evidence is hard to trust. This project is a Retrieval-Augmented Generation (RAG) app that solves both problems:

- You upload a whole set of PDFs in one go.
- The app splits them into chunks, converts the chunks into embeddings and stores them in **one shared vector index**, so a single question searches every document.
- The answer is generated **only from the retrieved text**, and the sources (`file - page`) are listed underneath.
- When two documents disagree, the app is designed to **say so** instead of quietly picking one.

## Key Features

| Feature | Description |
|---|---|
| Multi-PDF upload | Upload many PDFs at once and search across all of them |
| Single vector index | All documents are chunked and embedded into one store |
| Source citations | Every answer ends with the file name and page number, taken from chunk metadata |
| Conflict detection | The prompt tells the model to start with `CONFLICT` and explain each source when documents disagree; the app then shows a warning |
| Chunk-size hint | If one answer uses more than 4 different pages, the app suggests increasing the chunk size |
| Switchable embeddings | Choose Hugging Face `all-MiniLM-L6-v2` (local) or OpenAI `text-embedding-3-small` from the sidebar |
| Adjustable chunking | Chunk size and overlap sliders in the sidebar |
| FAISS with fallback | Uses FAISS when available, otherwise falls back to LangChain's `InMemoryVectorStore` (useful where Windows blocks native modules) |
| Optional tracing | LangSmith tracing can be switched on from the sidebar when an API key is present |
| Secure configuration | API keys are read from environment variables or a local `.env` file, never hard-coded |

## How It Works

```
 PDFs ──► PyPDFLoader ──► Text splitter ──► Embeddings ──► Vector store (FAISS)
 (upload)  (page by page)  (chunks + metadata)  (HF or OpenAI)        │
                                                                      ▼
 Answer + sources ◄── gpt-4o-mini ◄── Prompt with top 6 chunks ◄── Similarity search
```

1. **Load:** `PyPDFLoader` reads each PDF page by page.
2. **Label:** each page's metadata is set to the real file name and a 1-based page number.
3. **Split:** `RecursiveCharacterTextSplitter` cuts pages into chunks, and the metadata is carried into every chunk.
4. **Embed and store:** all chunks from all PDFs go into a single vector store.
5. **Retrieve:** the 6 chunks most similar to the question are found.
6. **Answer:** `gpt-4o-mini` (temperature 0) answers using only those chunks. The app lists the unique `file - page` sources below the answer.

## Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **Orchestration:** LangChain
- **Vector store:** FAISS (fallback: `InMemoryVectorStore`)
- **Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2` or OpenAI `text-embedding-3-small`
- **LLM:** OpenAI `gpt-4o-mini`
- **Observability (optional):** LangSmith

## Project Structure

```
multi-pdf-rag/
├── rag_app.py            # The entire application (Streamlit UI + RAG pipeline)
├── requirements.txt      # Python dependencies
├── .env                  # Local API keys (not committed to Git)
├── .gitignore            # Keeps .env and venv/ out of the repository
├── README.md             # Project documentation
├── sample_pdfs/          # (optional) example documents for testing
└── docs/
    └── screenshot.png    # Screenshot used in this README
```

Inside `rag_app.py` the code is organised in clear stages:

| Section | Purpose |
|---|---|
| Imports and `.env` loading | Load keys and choose FAISS or the in-memory fallback |
| Sidebar settings | Embedding model, chunk size, chunk overlap, LangSmith tracing |
| Step 1: Upload | Multi-file PDF uploader |
| Step 2: Build index | Load, label, split, embed and store all chunks |
| Step 3: Ask a question | Retrieve, build the prompt, answer, show sources and warnings |

## Getting Started

### Prerequisites

- Python 3.10 or newer
- An OpenAI API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project folder:

```env
OPENAI_API_KEY=your-openai-key

# Optional: enable LangSmith tracing
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=pdf-rag-assignment
```

### Run

```bash
streamlit run rag_app.py
```

### `requirements.txt`

```
streamlit
python-dotenv
langchain
langchain-openai
langchain-community
langchain-core
langchain-text-splitters
langchain-huggingface
sentence-transformers
pypdf
faiss-cpu
```

## Usage

1. Choose an embedding model in the sidebar.
2. Upload two or more PDFs.
3. Click **Build index** and wait for the chunk count.
4. Type a question and read the answer with its sources.

**Example**

```
Uploaded: patient_care_safety_manual.pdf, infection_control_policy.pdf, staff_handbook.pdf
Indexed 87 chunks using Hugging Face all-MiniLM-L6-v2 (local)

Q: How many leave days do new joiners get?
A: New joiners get 14 days of paid annual leave in their first year.
Sources: staff_handbook.pdf - page 4
```

*(The chunk count is an example; yours depends on the documents and chunk settings.)*

## Design Notes

- **Disagreeing documents:** conflict handling is done through the prompt, so the model, not the code, decides when sources disagree.
- **Page numbers:** `PyPDFLoader` counts pages from 0, so the app adds 1 to match what a reader sees in a PDF viewer.
- **Chunk size:** if one answer cites many different pages, the chunks are probably too small. The app flags this so the user can rebuild with a larger size.
- **Privacy:** with LangSmith tracing on, prompts, retrieved PDF text and model responses may be recorded. Leave tracing off for confidential documents.

## Limitations

- Scanned PDFs without a text layer are not supported (no OCR).
- Conflict detection depends on the language model following the prompt, so it is not guaranteed.
- The index lives in the Streamlit session and is rebuilt after a restart.
- Answers need an OpenAI API key. With Hugging Face embeddings, only the embedding step runs locally.
- On computers with strict application-control policies, native packages such as FAISS or the local Hugging Face model may be blocked. The app falls back to the in-memory store, and the OpenAI embedding option avoids local model files.

## Future Improvements

- OCR support for scanned PDFs
- Persistent index saved to disk
- Re-ranking and hybrid (keyword + vector) search
- Answer streaming and chat history
- Evaluation set to measure retrieval and answer quality

## License

MIT
