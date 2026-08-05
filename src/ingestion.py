from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_chunk_pdf(filepath):
    print(f"Loading PDF: {filepath}")
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_pdf("D:\\rag-project\\data\\policy.pdf")
    
    print("\n--- First chunk ---")
    print(chunks[0].page_content)
    
    print("\n--- Metadata ---")
    print(chunks[0].metadata)