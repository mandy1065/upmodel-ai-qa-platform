from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def embed_single_text(text: str):
    embeddings = get_embedding_model()
    vector = embeddings.embed_query(text)

    print(f"Text: {text[:50]}...")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 numbers: {vector[:5]}")

    return vector

if __name__ == "__main__":
    embed_single_text(
        "What is the refund policy for digital products?"
    )