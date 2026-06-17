# OmegaClaw (mettaclaw)

OmegaClaw is an agentic AI system implemented in **MeTTa**. Its repository is also
known as **mettaclaw**. Beyond basic tool use, OmegaClaw features embedding-based
long-term memory represented entirely in **MeTTa AtomSpace** format.

## Long-term memory

Long-term memory is deliberately maintained by the agent through MeTTa operations:

- `(remember string)` - add a memory item.
- `(query string)` - query related memories.
- `(episodes time)` - retrieve episodes around a point in time.
- `(pin string)` - add a message to the agent's own episodic trace.

Each memory item is stored as a triplet of `(timestamp, atom, embedding)`. The
agent remains flexible in choosing the specific representation, so different memory
formats can coexist in the same AtomSpace and other Hyperon components can operate
on the same memories.

## Tools

OmegaClaw implements an initial set of OpenClaw-like tools, including:

- Web search.
- File modification.
- Communication channels.
- Access to the operating-system shell and its associated tools.

## Design philosophy

The primary design criteria were simplicity of design, ease of prototyping, ease
of extension, and transparent implementation in MeTTa. The lean agent core
comprises approximately **200 lines of code**. OmegaClaw uses a token-efficient
agentic loop, enabling low-cost, long-running operation in domains that require
real-time learning and decision-making.

## Relationship to other components

- OmegaClaw **is implemented in** MeTTa.
- OmegaClaw **stores memory in** the AtomSpace.
- OmegaClaw memory items **contain** embeddings, enabling semantic recall.
