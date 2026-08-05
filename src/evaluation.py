from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from src.rag_pipeline import ask
from src.vectorstore import search_basic
from dotenv import load_dotenv
from src.synthesizer import load_test_data  # ← add this
from deepeval.evaluate.configs import AsyncConfig

load_dotenv()

test_data = load_test_data("test-data/synthetic_tests.json")  # ← load test data
test_data = test_data[:5]
def evaluate_rag():
    test_cases = []

    for item in test_data:
        question = item["question"]
        expected = item["expected"]

        docs = search_basic(question, k=4)
        context = [doc.page_content for doc in docs]

        answer = ask(question)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected,
            retrieval_context=context
        )
        test_cases.append(test_case)

    # ← inside function, same level as for loop
    metrics = [
        FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True),
        ContextualRelevancyMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True),
        ContextualPrecisionMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True),
        ContextualRecallMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True),
    ]

    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False)      # ← run one at a time
        
    )


if __name__ == "__main__":
    evaluate_rag()