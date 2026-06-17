# Advanced RAG vs Basic RAG - OpenCog / iCog Labs Corpus

A side-by-side implementation and evaluation of a **Basic RAG** baseline and a
feature-rich **Advanced RAG** system, built for the iCog Labs *Advanced RAG
Practical Assignment*. Both systems run on the same corpus and the same test
questions so the impact of each advanced technique can be measured directly.

What's implemented (all required features, plus two optional ones):

| Assignment item | Status | Where |
|---|---|---|
| Basic RAG (fixed chunks, embeddings, vector DB, top-k, LLM) | required | `src/basic_rag.py` |
| Advanced chunking: context-aware + hierarchical parent/child | required (>=2) | `src/chunking.py` |
| Query expansion + multi-query + merge/dedup (RRF) | required | `src/query_handling.py` |
| Re-ranking (cross-encoder) | optional/bonus | `src/rerank.py` |
| Corrective RAG (grade, rewrite/retry, refuse) | optional | `src/corrective.py` |
| GraphRAG write-up (3 conceptual questions) | required | `docs/GRAPHRAG.md` |
| 18 test questions across all 4 categories + comparison | required | `eval/`, `docs/COMPARISON.md` |

## Architecture

Basic RAG:

```
docs -> fixed-size chunks -> embeddings -> vector DB -> top-k -> LLM -> answer
```

Advanced RAG:

```
docs -> context-aware section splitting -> hierarchical parent/child chunks

question -> query expansion + multi-query -> vector retrieval -> RRF fuse + dedup
         -> cross-encoder re-rank
         -> corrective grading (if weak: rewrite & retry, else refuse)
         -> parent-section expansion -> LLM -> grounded answer with citations
```

## Tech stack

- **LLM:** pluggable `LLMProvider` interface (`src/llm.py`) - **Groq** (free, fast)
  or Google **Gemini** (free tier), selected with `LLM_PROVIDER`.
- **Embeddings:** local **sentence-transformers** (`BAAI/bge-small-en-v1.5`) - no
  API key, runs on CPU.
- **Re-ranker:** local cross-encoder (`ms-marco-MiniLM-L-6-v2`).
- **Vector DB:** self-contained, persistent **NumPy** vector store
  (`src/vectorstore.py`) - cosine similarity over a normalised embedding matrix, no
  external service. Swap-in Chroma/FAISS would touch only this one file.

## Setup

Requires **Python 3.12** (the local ML wheels target 3.12; 3.13/3.14 may lack
`torch` wheels).

```bash
# 1. create the environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# (torch CPU wheel: if the default is slow, use:
#  pip install torch --index-url https://download.pytorch.org/whl/cpu)

# 2. add a free API key
cp .env.example .env
#   then edit .env: set LLM_PROVIDER=groq and GROQ_API_KEY=...  (https://console.groq.com/keys)
#   (or LLM_PROVIDER=gemini and GEMINI_API_KEY=...)

# 3. build the indexes (Basic + Advanced)
python scripts/build_index.py

# 4. run the full Basic-vs-Advanced comparison
python scripts/run_eval.py            # add --judge llm for LLM-as-judge scoring
#   -> writes eval/results/results.json and docs/COMPARISON.md
```

## Try a single question

```bash
python scripts/ask.py --system both "How are PeTTa and MORK related?"
python scripts/ask.py --system advanced "How many employees does iCog Labs have?"  # should refuse
```

## Repository layout

```
data/raw/            the corpus (12 OpenCog/iCog markdown documents)
src/                 all pipeline modules (see the table above)
scripts/             build_index.py, run_eval.py, ask.py
eval/                test_questions.json, results/
docs/                GRAPHRAG.md (concept answers), COMPARISON.md (generated results)
```

## How the Advanced system fixes Basic-RAG failure modes

| Basic-RAG weakness | Advanced-RAG fix |
|---|---|
| Fixed chunks split headings/definitions mid-thought | Context-aware section splitting + hierarchical parent/child |
| An isolated chunk loses its context | Situating header prepended to each child + parent-section expansion |
| User wording differs from document wording (low recall) | Query expansion + multi-query + RRF fusion |
| Irrelevant chunks reach the LLM (low precision) | Cross-encoder re-ranking |
| Confident answers with weak evidence / hallucination | Corrective grading + refusal |

See `docs/COMPARISON.md` for the measured results and concrete win/fail examples.
