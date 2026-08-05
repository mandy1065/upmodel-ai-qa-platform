from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.embeddings import get_embedding_model
import chromadb
from chromadb.config import Settings

load_dotenv()

CHROMA_PATH = "D:\\rag-project\\chroma"


def get_chroma_client():
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))
    return client


def store_chunks(chunks):
    print("Creating vector store...")
    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="rag_collection"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def load_vectorstore():
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="rag_collection"
    )


def search_basic(query: str, k: int = 4):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def search_with_scores(query: str, k: int = 4):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    for doc, score in results:
        print(f"Score: {score:.3f} | {doc.page_content[:80]}...")
    return results


def search_diverse(query: str, k: int = 4):
    vectorstore = load_vectorstore()
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=5
    )
    return docs


if __name__ == "__main__":
    query = "Can I return a product?"

    print("\n=== Method 1: Basic Search ===")
    docs = search_basic(query)
    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}: {doc.page_content[:80]}...")

    print("\n=== Method 2: Search with Scores ===")
    search_with_scores(query)

    print("\n=== Method 3: MMR Diverse Search ===")
    search_diverse(query)