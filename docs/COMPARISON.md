# Basic RAG vs Advanced RAG - Comparison Results

Corpus: OpenCog Hyperon / MeTTa / iCog Labs documentation (18 test questions: 15 answerable, 3 unanswerable).

**Scoring:** answer correctness scored by an LLM-as-judge against the gold answer.

**Setup.** Both systems share the same corpus, embedding model (`BAAI/bge-small-en-v1.5`), vector store, and generation model, so the comparison isolates the retrieval techniques. *Basic* is the naive baseline: fixed-size chunks, single-query retrieval of the top-3 chunks by cosine similarity, no re-ranking. *Advanced* uses context-aware + hierarchical chunking, query expansion + multi-query with RRF de-duplication, cross-encoder re-ranking of a 12-candidate pool down to the best 6, corrective relevance grading, and parent-section expansion.

## Test questions

| ID | Category | Question | Gold answer |
|---|---|---|---|
| Q01 | factual | What does MeTTa stand for? | Meta Type Talk |
| Q02 | factual | In which city and country is iCog Labs located? | Addis Ababa, Ethiopia |
| Q03 | factual | What does the acronym MORK stand for? | MeTTa Optimal Reduction Kernel |
| Q04 | factual | Which minimum version of SWI-Prolog does PeTTa require? | SWI-Prolog 9.3.x or higher |
| Q05 | paraphrase | Which language is used to read from and write to the knowledge store in Hyperon, replacing the older Atomese? | MeTTa |
| Q06 | paraphrase | What mechanism decides which pieces of knowledge the system should focus its limited resources on, and how does it model that decision? | ECAN (Economic Attention Networks), which spreads attention values (STI/LTI) using an economic model where attention is a limited currency |
| Q07 | paraphrase | Whose PhD research are the core ideas behind the MOSES program learner based on? | Moshe Looks (2006 PhD thesis 'Competent Program Evolution') |
| Q08 | paraphrase | How does the OmegaClaw agent keep memories over long periods of time? | Embedding-based long-term memory stored in the AtomSpace as (timestamp, atom, embedding) triplets, managed with remember/query/episodes/pin operations |
| Q09 | relationship | The person who founded SingularityNET also co-founded which Ethiopian AI company, and in what city is that company based? | Ben Goertzel founded SingularityNET and co-founded iCog Labs, which is based in Addis Ababa |
| Q10 | relationship | How are PeTTa and MORK related? | PeTTa uses MORK (via the mork_ffi bindings) to provide high-performance MORK-based atom spaces |
| Q11 | relationship | metta-moses is written in which MeTTa implementation, and what abstract machine does that implementation run on? | metta-moses is written in MeTTaLog, which runs on the Warren Abstract Machine (WAM) via SWI-Prolog |
| Q12 | relationship | Which reasoning component draws inferences using the truth values attached to atoms in the AtomSpace? | PLN (Probabilistic Logic Networks) |
| Q13 | relationship | OmegaClaw keeps its long-term memory in a particular data structure; which high-performance kernel gives that same structure fast, scalable storage, and what does that kernel's name stand for? | OmegaClaw stores its long-term memory in the AtomSpace; MORK, which stands for MeTTa Optimal Reduction Kernel, gives the AtomSpace fast, scalable storage. |
| Q14 | relationship | metta-moses reimplements an earlier OpenCog program learner; what is that learner called and whose 2006 PhD thesis are its core ideas derived from? | metta-moses reimplements MOSES, whose core ideas are derived from Moshe Looks' 2006 PhD thesis 'Competent Program Evolution'. |
| Q15 | relationship | PeTTa implements a language whose programs run against a core knowledge store; which kernel accelerates that store for billions of atoms and what does its acronym stand for? | PeTTa implements MeTTa, which operates on the AtomSpace; MORK (MeTTa Optimal Reduction Kernel) accelerates that store so it scales to billions of atoms. |
| Q16 | unanswerable | How many full-time employees does iCog Labs currently have? | Not stated in the corpus - the system should refuse. |
| Q17 | unanswerable | What is the current market price of the SingularityNET (AGIX) token? | Not stated in the corpus - the system should refuse. |
| Q18 | unanswerable | In which exact year was the MeTTa language first officially released to the public? | Not stated in the corpus - the system should refuse. |

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
