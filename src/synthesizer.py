from deepeval.synthesizer import Synthesizer
from dotenv import load_dotenv
import json

load_dotenv()


def generate_test_data(pdf_path: str, num_questions: int = 10):
    print(f"Generating test data from: {pdf_path}")

    from src.ingestion import load_and_chunk_pdf
    chunks = load_and_chunk_pdf(pdf_path)

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    test_data = []
    questions_per_chunk = max(1, num_questions // len(chunks))

    for chunk in chunks[:num_questions]:
        prompt = f"""Read this text and generate ONE question and answer pair.

Text: {chunk.page_content}

Rules:
- Question must be answerable from this text only
- Answer must come directly from this text
- Be specific not vague

Return ONLY valid JSON like this exact format:
{{"question": "your question here", "expected": "your answer here"}}

Do not add any other text, just the JSON."""

        try:
            response = llm.invoke(prompt).content.strip()
            # clean response
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            item = json.loads(response)
            if "question" in item and "expected" in item:
                test_data.append(item)
                print(f"Generated: {item['question'][:60]}...")
        except Exception as e:
            print(f"Skipped chunk: {e}")

    return test_data


def save_test_data(test_data, output_path: str = "test-data/synthetic_tests.json"):
    # handle both list and synthesizer object
    if not isinstance(test_data, list):
        items = []
        for golden in test_data.synthetic_goldens:
            items.append({
                "question": golden.input,
                "expected": golden.expected_output
            })
        test_data = items

    with open(output_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Saved {len(test_data)} test cases to {output_path}")
    return test_data


def load_test_data(input_path: str = "test-data/synthetic_tests.json"):
    with open(input_path, "r") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test cases from {input_path}")
    return test_data


if __name__ == "__main__":
    import os
    os.makedirs("test-data", exist_ok=True)

    test_data = generate_test_data(
        pdf_path="D:\\rag-project\\data\\policy.pdf",
        num_questions=5
    )

    save_test_data(test_data)

    print("\n--- Generated Test Cases ---")
    for i, item in enumerate(test_data):
        print(f"\nTest {i+1}:")
        print(f"Q: {item['question']}")
        print(f"A: {item['expected']}")