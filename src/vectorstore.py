from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.embeddings import get_embedding_model

load_dotenv()

CHROMA_PATH = "./chroma"


def store_chunks(chunks):
    print("Creating vector store...")
    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="rag_docs"
    )
    
    print(f"Stored {len(chunks)} chunks")
    return vectorstore


def load_vectorstore():
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="rag_docs"
    )


def search_basic(query: str, k: int = 4):
    return load_vectorstore().similarity_search(query, k=k)


def search_with_scores(query: str, k: int = 4):
    results = load_vectorstore().similarity_search_with_score(query, k=k)
    for doc, score in results:
        print(f"Score: {score:.3f} | {doc.page_content[:80]}...")
    return results


def search_diverse(query: str, k: int = 4):
    return load_vectorstore().max_marginal_relevance_search(
        query, k=k, fetch_k=5
    )