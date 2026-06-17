# Basic RAG vs Advanced RAG - Comparison Results

Corpus: OpenCog Hyperon / MeTTa / iCog Labs documentation (18 test questions: 15 answerable, 3 unanswerable).

**Scoring:** answer correctness scored by an LLM-as-judge against the gold answer.

**Setup.** Both systems share the same corpus, embedding model (`BAAI/bge-small-en-v1.5`), vector store, and generation model, so the comparison isolates the retrieval techniques. *Basic* is the naive baseline: fixed-size chunks, single-query retrieval of the top-3 chunks by cosine similarity, no re-ranking. *Advanced* uses context-aware + hierarchical chunking, query expansion + multi-query with RRF de-duplication, cross-encoder re-ranking of a 12-candidate pool down to the best 6, corrective relevance grading, and parent-section expansion.

## Summary metrics

| Metric | Basic RAG | Advanced RAG |
|---|---|---|
| Retrieval recall (answerable) | 100% | 100% |
| Answer accuracy (answerable) | 87% | 100% |
| Grounded / supported rate | 87% | 100% |
| Correct refusal on unanswerable | 100% | 100% |
| Hallucination on unanswerable (lower=better) | 0% | 0% |
| Overall correct | 89% | 100% |

## Per-question results

| ID | Category | Basic correct | Adv correct | Basic grounded | Adv grounded | Basic hit | Adv hit |
|---|---|---|---|---|---|---|---|
| Q01 | factual | yes | yes | yes | yes | True | True |
| Q02 | factual | yes | yes | yes | yes | True | True |
| Q03 | factual | yes | yes | yes | yes | True | True |
| Q04 | factual | yes | yes | yes | yes | True | True |
| Q05 | paraphrase | yes | yes | yes | yes | True | True |
| Q06 | paraphrase | yes | yes | yes | yes | True | True |
| Q07 | paraphrase | yes | yes | yes | yes | True | True |
| Q08 | paraphrase | yes | yes | yes | yes | True | True |
| Q09 | relationship | yes | yes | yes | yes | True | True |
| Q10 | relationship | yes | yes | yes | yes | True | True |
| Q11 | relationship | yes | yes | yes | yes | True | True |
| Q12 | relationship | yes | yes | yes | yes | True | True |
| Q13 | relationship | no | yes | no | yes | True | True |
| Q14 | relationship | no | yes | no | yes | True | True |
| Q15 | relationship | yes | yes | yes | yes | True | True |
| Q16 | unanswerable | yes | yes | yes | yes | None | None |
| Q17 | unanswerable | yes | yes | yes | yes | None | None |
| Q18 | unanswerable | yes | yes | yes | yes | None | None |

## Examples where Advanced RAG did better

### Q13 - OmegaClaw keeps its long-term memory in a particular data structure; which high-performance kernel gives that same structure fast, scalable storage, and what does that kernel's name stand for?
- **Category:** relationship  (accuracy win)
- **Basic answer:** OmegaClaw keeps its long-term memory in a MeTTa AtomSpace format. The high-performance kernel that gives this structure fast, scalable storage is called Hyperon, and its name stands for "Hybrid Reasoning Engine for OpenCog". 

The context mentions that different memory formats can coexist in the same AtomSpace and other Hyperon components can operate on the same memories [3].
- **Advanced answer:** OmegaClaw keeps its long-term memory in the AtomSpace data structure. The high-performance kernel that gives this structure fast, scalable storage is MORK. MORK's name stands for MeTTa Optimal Reduction Kernel [4].
- **Why advanced was better:** Multi-query retrieval + re-ranking gathered the spread-out evidence needed to connect the facts.

### Q14 - metta-moses reimplements an earlier OpenCog program learner; what is that learner called and whose 2006 PhD thesis are its core ideas derived from?
- **Category:** relationship  (accuracy win)
- **Basic answer:** I cannot answer this question based on the provided documents. The available evidence does not contain the necessary information.
- **Advanced answer:** The earlier OpenCog program learner that metta-moses reimplements is called MOSES. Its core ideas are derived from Moshe Looks' 2006 PhD thesis, "Competent Program Evolution" (Washington University in St. Louis, Missouri) [1, 3].
- **Why advanced was better:** Multi-query retrieval + re-ranking gathered the spread-out evidence needed to connect the facts.

## Examples where Advanced RAG still failed

_(Advanced RAG answered every question correctly on this run.)_
