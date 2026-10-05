# Multi-PDF RAG app with sources
# Run with:  streamlit run rag_app.py

import os  # lets us read the API key from an environment variable and delete temp files
import tempfile  # lets us save uploaded PDFs to a temporary file on disk
from importlib import import_module
from pathlib import Path
from dotenv import load_dotenv  # lets us read API keys from a .env file

load_dotenv(Path(__file__).resolve().parent / ".env")

import streamlit as st  # Streamlit builds the web page for us
from langchain_community.document_loaders import PyPDFLoader  # reads a PDF page by page
from langchain_text_splitters import RecursiveCharacterTextSplitter  # cuts pages into smaller chunks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:  # use FAISS when its native module is allowed on this computer
    import_module("faiss")
    from langchain_community.vectorstores import FAISS
    VectorStore = FAISS
except ImportError:  # fall back if FAISS is missing or blocked by Windows
    from langchain_core.vectorstores import InMemoryVectorStore
    VectorStore = InMemoryVectorStore

st.title("Ask your PDFs")  # big title at the top of the page

# Stop the app early if the API key is missing
if not os.getenv("OPENAI_API_KEY"):  # check that the environment variable exists
    st.error("Please set OPENAI_API_KEY and restart the app.")  # tell the user what to do
    st.stop()  # stop running the rest of the page

# ---------- Sidebar settings ----------
st.sidebar.caption(f"Vector store: {VectorStore.__name__}")  # show which vector store is being used
embed_choice = st.sidebar.selectbox(
    "Embedding model",
    [
        "Hugging Face all-MiniLM-L6-v2 (local, no HF API key)",
        "OpenAI text-embedding-3-small",
    ],
)
chunk_size = st.sidebar.slider("Chunk size", 300, 2000, 800, 100)  # how many characters in each chunk
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 150, 50)  # characters shared between neighbouring chunks

# LangChain automatically sends traces to LangSmith when these variables are set.
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
trace_enabled = st.sidebar.checkbox(
    "Enable LangSmith tracing",
    value=bool(langsmith_api_key),
    disabled=not bool(langsmith_api_key),
)
if langsmith_api_key and trace_enabled:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "pdf-rag-assignment")
    st.sidebar.caption(
        f"Tracing to LangSmith project: {os.environ['LANGSMITH_PROJECT']}. "
        "Prompts, retrieved PDF text, and model responses may be recorded."
    )
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    if not langsmith_api_key:
        st.sidebar.caption("Add LANGSMITH_API_KEY to .env to enable tracing.")

# ---------- Step 1: upload PDFs ----------
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)  # allow many PDFs at once

# ---------- Step 2: build the index ----------
if st.button("Build index") and files:  # run only when the button is clicked and files are uploaded
    chunks = []  # this list will hold the chunks from ALL the PDFs
    splitter = RecursiveCharacterTextSplitter(  # create the tool that cuts text into chunks
        chunk_size=chunk_size,  # use the chunk size chosen in the sidebar
        chunk_overlap=chunk_overlap,  # use the overlap chosen in the sidebar
    )

    for f in files:  # go through each uploaded PDF one by one
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:  # make a temporary file
            tmp.write(f.getvalue())  # write the uploaded PDF's bytes into it
            path = tmp.name  # remember where the temporary file is saved

        pages = PyPDFLoader(path).load()  # read the PDF; each page becomes one document
        os.remove(path)  # delete the temporary file, we no longer need it

        for page in pages:  # go through every page of this PDF
            page.metadata["source"] = f.name  # replace the temp path with the real file name
            page.metadata["page"] = page.metadata["page"] + 1  # PyPDFLoader starts counting at 0, so add 1

        chunks.extend(splitter.split_documents(pages))  # cut pages into chunks (metadata is kept) and add to the list

    if embed_choice.startswith("OpenAI"):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.embed_name = "OpenAI text-embedding-3-small"
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.session_state.embed_name = "Hugging Face all-MiniLM-L6-v2 (local)"
    st.session_state.store = VectorStore.from_documents(chunks, embeddings)  # embed all chunks into ONE vector database
    st.session_state.count = len(chunks)  # remember how many chunks we indexed

# Show the chunk count whenever an index exists
if "store" in st.session_state:  # check that the index has been built
    st.success(f"Indexed {st.session_state.count} chunks using {st.session_state.embed_name}")  # show chunk count and embedding model

# ---------- Step 3: ask a question ----------
question = st.text_input(
    "Ask a question about your PDFs",
    help="Enter exit or quit to stop processing the current chat input.",
).strip()

if question and "store" in st.session_state:  # run only if there is a question and an index
    docs = st.session_state.store.similarity_search(question, k=6)  # find the 6 most similar chunks

    context = ""  # start with an empty text that will hold all the chunks
    for d in docs:  # go through each retrieved chunk
        context += f"[{d.metadata['source']}, page {d.metadata['page']}]\n{d.page_content}\n\n"  # add chunk with its label

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say you could not find it.
If the sources give different answers, start your reply with the word CONFLICT
and explain what each file and page says. Do not quietly pick one.

Context:
{context}

Question: {question}"""  # the instructions we send to the AI, with our chunks and question filled in

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # create the chat model (temperature 0 = less random)
    answer = llm.invoke(prompt).content  # ask the model and take the text of its reply

    st.write(f"**Q:** {question}")  # show the question
    if answer.startswith("CONFLICT"):  # check whether the AI found disagreeing sources
        st.warning("Your documents disagree on this question:")  # show a warning box
    st.write(f"**A:** {answer}")  # show the answer

    pairs = []  # list that will hold each unique (file, page) source
    for d in docs:  # go through each retrieved chunk again
        pair = (d.metadata["source"], d.metadata["page"])  # get its file name and page number
        if pair not in pairs:  # skip duplicates so each page is listed once
            pairs.append(pair)  # add the new source to the list

    sources_text = " | ".join(f"{name} - page {page}" for name, page in pairs)  # join sources into one line
    st.write(f"**Sources:** {sources_text}")  # show the sources under the answer

    if len(pairs) > 4:  # many different pages usually means chunks are too small
        st.info("This answer uses more than 4 pages. Try a bigger chunk size and rebuild the index.")  # give a hint
elif question:  # a question was typed but no index exists yet
    st.info("Upload your PDFs and click Build index first.")  # remind the user what to do