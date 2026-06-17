# GraphRAG & Knowledge-Graph Retrieval

This document answers the three conceptual questions in Section 4 of the
assignment, and sketches how hybrid (vector + knowledge-graph) retrieval could
extend the Advanced system.

**Background reading used:**
- Microsoft Research, *"From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization"* (Edge et al., 2024) - the GraphRAG paper.
- Microsoft GraphRAG documentation (https://microsoft.github.io/graphrag/).
- Neo4j, *"GraphRAG: The Practical Guide"* and the LlamaIndex Knowledge-Graph
  retriever docs.

---

## 1. How Knowledge-Graph retrieval differs from vector retrieval

| Aspect | Vector retrieval | Knowledge-Graph retrieval |
|---|---|---|
| Unit of knowledge | Text chunks (passages) | Entities (nodes) and relationships (edges) |
| Matching method | Semantic similarity in embedding space (nearest neighbours) | Graph pattern matching / traversal of explicit relationships |
| What it is good at | "Find text that *talks about* X" | "Find how X is *connected to* Y", multi-hop chains |
| Multi-hop reasoning | Weak - each chunk is retrieved independently; the model must stitch facts together, and the bridging chunk may not be similar to the query | Strong - you can literally walk edges: X -> relation -> Y -> relation -> Z |
| Failure mode | Returns plausible-looking but unrelated passages; misses facts split across documents | Limited by extraction quality; misses facts never extracted as triples |
| Representation | Opaque float vectors | Human-readable, inspectable facts |

**In short:** vector search answers *"what text is similar to my question?"* while
graph search answers *"what entities are related to the things in my question, and
how?"* Vector search is similarity-based and local to a chunk; graph search is
structure-based and can connect facts that live in *different* documents.

A concrete example from this corpus:

> *"metta-moses is written in which MeTTa implementation, and what abstract machine
> does that implementation run on?"*

Pure vector search tends to retrieve the `metta-moses` passage (which says it is
"written in MeTTaLog") but may not retrieve the *separate* `metta-wam` passage that
says "MeTTaLog runs on the Warren Abstract Machine." The graph can traverse:

```
metta-moses --IS_WRITTEN_IN--> MeTTaLog --RUNS_ON--> Warren Abstract Machine
```

and surface the whole chain as evidence.

---

## 2. Entities and relationships in this dataset

The corpus is about the OpenCog Hyperon AGI ecosystem. Analysing the documents,
the entities (by type) and relationships are:

- **Person** - Ben Goertzel, Getnet Aseffa, Moshe Looks
- **Organization** - iCog Labs, SingularityNET, TrueAGI (trueagi-io), OpenCog Foundation
- **Project / Software** - OpenCog Hyperon, OpenCog Classic, MORK, PeTTa, MeTTaLog
  (metta-wam), metta-moses, MOSES, OmegaClaw (mettaclaw), bio-semantic-parser
- **Language** - MeTTa, Prolog (SWI-Prolog), Atomese, combo
- **Technology / Concept** - AtomSpace, PLN, ECAN, FAISS, Warren Abstract Machine,
  metagraph, cognitive synergy
- **Location** - Addis Ababa, Ethiopia

Representative relationships (edges):

```
Ben Goertzel        --FOUNDED-->            SingularityNET
Ben Goertzel        --CO_FOUNDED-->         iCog Labs
iCog Labs           --LOCATED_IN-->         Addis Ababa
iCog Labs           --CONTRIBUTES_TO-->     OpenCog Hyperon
OpenCog Hyperon     --SUCCESSOR_OF-->       OpenCog Classic
OpenCog Hyperon     --USES-->               MeTTa
MeTTa               --OPERATES_ON-->        AtomSpace
PLN                 --REASONS_OVER-->       AtomSpace
ECAN                --ALLOCATES_ATTENTION-->AtomSpace
PeTTa               --IMPLEMENTS-->         MeTTa
PeTTa               --USES-->               MORK
PeTTa               --USES-->               FAISS
MORK                --BACKEND_FOR-->        AtomSpace
MeTTaLog            --RUNS_ON-->            Warren Abstract Machine
metta-moses         --WRITTEN_IN-->         MeTTaLog
metta-moses         --REIMPLEMENTS-->       MOSES
MOSES               --DERIVED_FROM-->       Moshe Looks (thesis)
OmegaClaw           --STORES_MEMORY_IN-->   AtomSpace
SingularityNET      --FUNDS-->              OpenCog Hyperon
```

---

## 3. How graph retrieval improves relationship-based questions

Relationship and multi-hop questions are exactly where naive vector RAG is weakest,
because the answer is not contained in any *single* similar chunk - it is spread
across several documents and must be **connected**. Graph retrieval helps by:

1. **Following explicit edges instead of guessing from similarity.** For
   "What is the relationship between PeTTa and MORK?", the graph returns the exact
   edge `PeTTa --USES--> MORK`, rather than hoping both terms co-occur in one chunk.

2. **Multi-hop traversal.** For "The founder of SingularityNET co-founded which
   Ethiopian company, and where is it based?", the graph walks
   `Ben Goertzel --CO_FOUNDED--> iCog Labs --LOCATED_IN--> Addis Ababa`, joining two
   facts that live in two different documents.

3. **Connecting facts across documents.** Vector chunks are retrieved
   independently; the graph encodes cross-document links so a single traversal can
   gather all the bridging facts.

4. **Precise, inspectable evidence.** The returned facts are short, unambiguous
   triples, which reduces the chance the generator hallucinates the connection.

---

## 4. Hybrid retrieval - conceptual design (Optional/Bonus)

This project implements the **vector** side of the Advanced pipeline (multi-query
+ RRF + cross-encoder re-ranking + corrective grading). A natural bonus extension
is to add a knowledge-graph retriever and run the two **in parallel**:

```
question
  - vector retrieval (multi-query + RRF + re-rank)  -> relevant passages
  - knowledge-graph retrieval (entity match + traversal) -> relationship facts
        |
        v
  combine evidence -> corrective grading -> grounded answer
```

- **Vector side** supplies rich descriptive context (definitions, explanations).
- **Graph side** supplies the explicit relationships needed for multi-hop questions.
- The two evidence sets would be merged and de-duplicated, graded by the corrective
  stage, and passed to the generator together - grounding the answer in *both*
  similar text and explicit structure.

This mirrors the "Hybrid retrieval: Vector search + Knowledge Graph" slide from the
training: *vector search finds similar text; graph search follows relationships.*
In practice it would be built with a triple store or an in-process graph
(e.g. NetworkX), with edges extracted by an LLM pass over the corpus.
