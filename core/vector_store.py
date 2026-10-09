from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # older setups
    from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    # Remove the previous video's data so answers never mix two videos.
    try:
        Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        ).delete_collection()
    except Exception:
        pass

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )


def load_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
