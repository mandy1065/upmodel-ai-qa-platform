import streamlit as st
import os
import shutil
import gc
import time
from src.ingestion import load_and_chunk_pdf
from src.vectorstore import store_chunks, search_diverse
from src.rag_pipeline import ask
from src.synthesizer import generate_test_data, save_test_data
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.evaluate.configs import AsyncConfig
from dotenv import load_dotenv
from deepeval.evaluate.configs import CacheConfig
load_dotenv()

# ── PATHS ──────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma"
UPLOAD_PATH = "./data"
TEST_DATA_PATH = "test-data/synthetic_tests.json"

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UpModel — AI QA Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #2d3748;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 1rem;
        color: #a0aec0;
        margin-top: 0.3rem;
    }
    .brand-tag {
        display: inline-block;
        background: #4f46e5;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-pass { border-top: 3px solid #48bb78; }
    .metric-fail { border-top: 3px solid #fc8181; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        padding: 0.5rem 0;
        border-bottom: 1px solid #2d3748;
        margin-bottom: 1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="brand-title">🛡️ UpModel — AI QA Evaluation Platform</p>
    <p class="brand-subtitle">Protecting AI products from hallucination, drift & silent failure</p>
</div>
""", unsafe_allow_html=True)


# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def index_pdf(pdf_path: str) -> int:
    gc.collect()
    time.sleep(1)

    # delete corrupted chroma
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        time.sleep(2)

    # create fresh empty folder
    os.makedirs(CHROMA_PATH, exist_ok=True)

    chunks = load_and_chunk_pdf(pdf_path)
    store_chunks(chunks)
    return len(chunks)
    gc.collect()
    time.sleep(1)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        time.sleep(1)
    chunks = load_and_chunk_pdf(pdf_path)
    store_chunks(chunks)
    return len(chunks)


def run_evaluation(test_data: list, selected_metrics: list):
    test_cases = []
    progress = st.progress(0)
    status = st.empty()

    for i, item in enumerate(test_data):
        question = item["question"]
        expected = item["expected"]
        status.text(f"⚙️ Running test {i+1}/{len(test_data)}: {question[:60]}...")
        progress.progress((i+1) / len(test_data))

        docs = search_diverse(question, k=2)
        context = [doc.page_content[:300] for doc in docs]
        answer = ask(question)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected,
            retrieval_context=context
        )
        test_cases.append(test_case)

    status.text("📊 Evaluating with DeepEval metrics...")
    progress.progress(1.0)

    metric_map = {
        "Faithfulness": FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini", include_reason=False),
        "Answer Relevancy": AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", include_reason=False),
        "Contextual Relevancy": ContextualRelevancyMetric(threshold=0.7, model="gpt-4o-mini", include_reason=False),
        "Contextual Precision": ContextualPrecisionMetric(threshold=0.7, model="gpt-4o-mini", include_reason=False),
        "Contextual Recall": ContextualRecallMetric(threshold=0.7, model="gpt-4o-mini", include_reason=False),
    }
    metrics = [metric_map[m] for m in selected_metrics if m in metric_map]

    results = evaluate(
    
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False)
    )

    status.empty()
    progress.empty()
    return test_cases, metrics, results


def show_results(test_cases, metrics):
    st.markdown('<p class="section-header">📊 Overall Metric Scores</p>', unsafe_allow_html=True)

    cols = st.columns(len(metrics))
    for i, (col, metric) in enumerate(zip(cols, metrics)):
        scores = []
        for tc in test_cases:
            metric.measure(tc)
            scores.append(metric.score)

        avg = sum(scores) / len(scores) if scores else 0
        pass_rate = sum(1 for s in scores if s >= 0.7) / len(scores) * 100 if scores else 0
        name = metric.__class__.__name__.replace("Metric", "")
        card_class = "metric-pass" if avg >= 0.7 else "metric-fail"
        icon = "✅" if avg >= 0.7 else "❌"

        with col:
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <div style="font-size:1.6rem">{icon}</div>
                <div style="font-size:0.8rem;color:#a0aec0;margin:0.3rem 0">{name}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{'#48bb78' if avg >= 0.7 else '#fc8181'}">{avg:.2f}</div>
                <div style="font-size:0.75rem;color:#718096">{pass_rate:.0f}% pass rate</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="section-header">📋 Per Test Case Results</p>', unsafe_allow_html=True)

    for i, tc in enumerate(test_cases):
        with st.expander(f"Test {i+1}: {tc.input[:70]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**❓ Question**")
                st.info(tc.input)
                st.markdown("**🤖 RAG Answer**")
                st.success(tc.actual_output)
            with col2:
                st.markdown("**✅ Expected Answer**")
                st.warning(tc.expected_output)

            st.markdown("**📊 Metric Scores**")
            metric_cols = st.columns(len(metrics))
            for j, (mc, metric) in enumerate(zip(metric_cols, metrics)):
                metric.measure(tc)
                passed = metric.score >= 0.7
                with mc:
                    st.metric(
                        label=metric.__class__.__name__.replace("Metric", ""),
                        value=f"{metric.score:.2f}",
                        delta="PASS ✅" if passed else "FAIL ❌"
                    )

    st.divider()


def run_analysis(uploaded_file, num_questions: int, selected_metrics: list):
    os.makedirs(UPLOAD_PATH, exist_ok=True)
    pdf_path = os.path.join(UPLOAD_PATH, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("📚 Indexing PDF into ChromaDB..."):
        num_chunks = index_pdf(pdf_path)
    st.success(f"✅ Indexed **{num_chunks} chunks** from `{uploaded_file.name}`")

    with st.spinner("🧪 Auto-generating test cases..."):
        os.makedirs("test-data", exist_ok=True)
        test_data = generate_test_data(pdf_path=pdf_path, num_questions=num_questions)
        test_data = save_test_data(test_data, TEST_DATA_PATH)

    if not test_data:
        st.error("❌ Could not generate test cases. Try a different PDF.")
        return

    st.success(f"✅ Generated **{len(test_data)} test cases** automatically")

    st.markdown('<p class="section-header">📋 Auto-Generated Test Cases</p>', unsafe_allow_html=True)
    for i, item in enumerate(test_data):
        with st.expander(f"Test {i+1}: {item['question'][:65]}..."):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Expected:** {item['expected']}")

    st.divider()

    st.markdown('<p class="section-header">⚙️ Running Evaluation Pipeline</p>', unsafe_allow_html=True)
    test_cases, metrics, results = run_evaluation(test_data, selected_metrics)

    if not test_cases:
        st.error("❌ Evaluation failed. Reduce test cases and try again.")
        return

    show_results(test_cases, metrics)


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():

    # ── SIDEBAR ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Evaluation Settings")
        num_questions = st.slider(
            "Test cases to generate",
            min_value=2, max_value=8, value=2,
            help="More = better coverage, slower evaluation"
        )

        st.markdown("### 📊 Select Metrics")
        metrics_options = [
            "Faithfulness",
            "Answer Relevancy",
            "Contextual Relevancy",
            "Contextual Precision",
            "Contextual Recall"
        ]
        selected_metrics = []
        for m in metrics_options:
            default = m in ["Faithfulness", "Answer Relevancy"]
            if st.checkbox(m, value=default):
                selected_metrics.append(m)

        st.divider()


    # ── UPLOAD SECTION ───────────────────────────────────────────────────
    st.markdown("### 📄 Upload Document")
    st.caption("Drop any PDF — policy, contract, handbook, knowledge base")

    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success(f"✅ Ready: `{uploaded_file.name}` ({uploaded_file.size // 1024} KB)")

        if not selected_metrics:
            st.warning("⚠️ Please select at least one metric in the sidebar.")
        else:
            clicked = st.button(
                "🚀 Run Full AI QA Analysis",
                type="primary",
                use_container_width=True
            )
            if clicked:
                run_analysis(uploaded_file, num_questions, selected_metrics)

    else:
        # ── HOW IT WORKS (shown when no file uploaded) ───────────────
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 How It Works")
            st.markdown("""
            | Step | What happens |
            |---|---|
            | **1. Upload** | Any PDF document |
            | **2. Index** | Chunks + embeds into ChromaDB |
            | **3. Generate** | Auto-creates test cases via AI |
            | **4. Evaluate** | Runs RAG + DeepEval metrics |
            | **5. Report** | Pass/fail with root cause |
            """)

        with col2:
            st.markdown("### 📊 Metric Guide")
            st.markdown("""
            - 🟢 **≥ 0.7** = Pass (industry standard)
            - 🔴 **< 0.7** = Fail → fix needed
            - **Faithfulness** → hallucination check
            - **Answer Relevancy** → on-topic check
            - **Contextual Relevancy** → retrieval quality
            - **Contextual Precision** → ranking quality
            - **Contextual Recall** → coverage check
            """)


if __name__ == "__main__":
    main()