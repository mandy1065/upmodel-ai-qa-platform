from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from src.vectorstore import search_basic 


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def build_prompt(question: str, context_docs: list) -> str:
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Answer using ONLY the context below.
Do not use any outside knowledge.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
"""
    return prompt


def ask(question: str) -> str:
    print(f"\nQuestion: {question}")
    
    # STEP 1 — retrieve relevant chunks
    docs = search_basic(question, k=4)
    
    # STEP 2 — build prompt
    prompt = build_prompt(question, docs)
    
    # STEP 3 — get answer from LLM
    answer = llm.invoke(prompt).content
    
    print(f"Answer: {answer}")
    return answer


if __name__ == "__main__":
    questions = [
        "Can I return a product?",
        "What is the handling charge for returns?",
        "How many days do I have to report damaged goods?"
    ]
    
    for question in questions:
        ask(question)
        print("-" * 50)