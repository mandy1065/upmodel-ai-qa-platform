# 🛡️ UpModel — AI QA Evaluation Platform

> **Protecting AI products from hallucination, drift & silent failure.**
> Built by [UpModel](https://www.upmodel.app) — Canada's AI QA & LLM Evaluation Specialists.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![DeepEval](https://img.shields.io/badge/DeepEval-4.1-green?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚨 The Problem

Conventional QA is built for deterministic software — same input, same output, always.

**AI systems are probabilistic. The rules have changed.**

| Traditional QA | AI / LLM QA |
|---|---|
| Pass/fail is binary | Quality is graded — thresholds, not hard rules |
| Static tests stay valid | Model drift means yesterday's pass can fail tomorrow |
| Bugs have stack traces | Hallucination = confident wrong answer, no trace |
| Security is a separate workstream | Prompt injection is a core QA concern from day one |

---

## ✅ What This Platform Does

Upload any AI-facing document and this platform automatically:

```
📄 Upload PDF
    ↓
🔪 Chunk + Embed into Vector Store
    ↓
🧪 Auto-Generate Test Cases (zero manual writing)
    ↓
🤖 Run RAG Pipeline on each test case
    ↓
📊 Evaluate across 5 industry-standard metrics
    ↓
📋 Show pass/fail results with root cause explanations
```

---

## 📊 Evaluation Metrics

| Metric | What it checks | Failure means |
|---|---|---|
| **Faithfulness** | Did the LLM hallucinate? | Fix your prompt constraints |
| **Answer Relevancy** | Did it answer the question asked? | Fix your prompt template |
| **Contextual Relevancy** | Were the right chunks retrieved? | Fix chunking or embedding |
| **Contextual Precision** | Best chunks ranked first? | Add a reranker |
| **Contextual Recall** | All needed info captured? | Increase k value |

---

## 🏆 Case Study — UpModel Platform

**Context:** AI Career Intelligence Platform built on RAG + Claude API — Career Gap Analyzer & Resume Scanner tools.

| What We Tested | Issues Found | Outcome |
|---|---|---|
| RAG retrieval accuracy | Context window overflow → truncated recommendations | DeepEval CI pipeline integrated |
| LLM hallucination scoring | 2 hallucination instances in salary range outputs | Faithfulness: **0.61 → 0.89** |
| Prompt injection vulnerability | Edge-case bypassing role restrictions | Attack vectors documented & patched |
| Answer faithfulness vs. source grounding | Faithfulness score below threshold | **73% reduction** in hallucination rate |
| Missing fallback handling | Null resume parse with no fallback | Zero regression failures in 3 release cycles |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit UI                   │
│    Upload PDF → Run Analysis → See Results  │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │   src/ingestion   │  PyPDF + LangChain chunking
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  src/embeddings   │  OpenAI text-embedding-3-small
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │ src/vectorstore   │  ChromaDB similarity + MMR search
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │ src/synthesizer   │  DeepEval auto test generation
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │ src/rag_pipeline  │  GPT-4o-mini answer generation
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │ src/evaluation    │  DeepEval 5-metric scoring
         └───────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key

### Setup

```bash
# 1. Clone
git clone https://github.com/yourusername/upmodel-ai-qa-platform
cd upmodel-ai-qa-platform

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env → add your OPENAI_API_KEY

# 5. Run
streamlit run app.py
```

---

## 📁 Project Structure

```
upmodel-ai-qa-platform/
│
├── app.py                 ← Streamlit web UI
├── requirements.txt       ← Pinned dependencies
├── .env.example           ← API key template (safe to commit)
├── .gitignore             ← Protects .env and secrets
│
└── src/
    ├── ingestion.py       ← PDF loading + chunking
    ├── embeddings.py      ← OpenAI embedding model
    ├── vectorstore.py     ← ChromaDB storage + 3 search methods
    ├── rag_pipeline.py    ← RAG answer generation
    ├── synthesizer.py     ← Auto test case generation
    └── evaluation.py      ← DeepEval 5-metric evaluation
```

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Document Processing | LangChain + PyPDF | PDF loading, chunking, RAG chain |
| Vector Storage | ChromaDB | Embedding storage + similarity search |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim semantic vectors |
| LLM | GPT-4o-mini | Answer generation + evaluation judge |
| Evaluation | DeepEval 4.1 | 5 RAG quality metrics |
| Test Generation | DeepEval Synthesizer | Auto Q&A pair generation |
| UI | Streamlit | Web interface |

---

## 🎯 Use Cases

- **AI Startups** — test RAG features before shipping to users
- **Enterprise Teams** — validate AI outputs meet quality and compliance standards
- **QA Consultants** — deliver structured AI evaluation reports to clients
- **Compliance Teams** — ensure AI answers are grounded in approved source documents
- **Product Teams** — catch hallucinations before they reach your users

---

## 💼 About UpModel

We are Canada-based AI QA specialists helping companies protect their AI products.

**Services:**

| | AI Eval Audit | Ongoing Retainer |
|---|---|---|
| **Price** | CAD $1,500–$3,000 | CAD $800–$2,000/mo |
| **Timeline** | 2–4 weeks | Monthly |
| **Includes** | Hallucination scoring, RAG assessment, prompt injection scan, fix recommendations | Monthly eval reports, release QA, Slack support, quarterly strategy call |

**Book a free 30-minute discovery call:**
🌐 [www.upmodel.app](https://www.upmodel.app)
📧 Contact via website
🔗 [Manjinder Dayal — LinkedIn](https://linkedin.com)

---

## 📄 License

MIT License — open source, free to use and adapt.

---

*Built with ❤️ by UpModel — AI QA & LLM Evaluation Specialists, Ontario, Canada.*