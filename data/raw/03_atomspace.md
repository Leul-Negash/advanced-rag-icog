# AtomSpace

The AtomSpace is the central knowledge-representation store of OpenCog and OpenCog
Hyperon. It is a weighted, labeled hypergraph - more precisely a *metagraph* - in
which all knowledge is stored as objects called **atoms**.

## Atoms: Nodes and Links

There are two basic kinds of atoms:

- **Nodes** - named entities, such as a `ConceptNode` representing "cat" or a
  `PredicateNode` representing a relationship type.
- **Links** - typed connections between atoms. Because links can connect other
  links (not just nodes), the AtomSpace is a metagraph rather than an ordinary
  graph. For example, an `InheritanceLink` can express that "cat is an animal."

Atoms can carry **values**, including **truth values** (degrees of belief) and
**attention values** (importance). Truth values are used by PLN for reasoning,
and attention values are managed by ECAN.

## Why a metagraph

A metagraph allows relationships to themselves be the subject of other
relationships. This makes it possible to represent higher-order knowledge, such
as statements about statements, rules about rules, and context-dependent truth.
Ordinary knowledge graphs (subject-predicate-object triples) cannot express this
directly.

## AtomSpace in Hyperon

In OpenCog Hyperon, the AtomSpace is manipulated using the MeTTa language rather
than the older Atomese. High-performance backends such as MORK provide fast
storage and pattern-matching over very large AtomSpaces. Applications such as the
OmegaClaw agent store their long-term memory directly as atoms in the AtomSpace.

## Relationship to other components

- **MeTTa** reads from and writes to the AtomSpace.
- **PLN** reasons over the truth values of atoms in the AtomSpace.
- **ECAN** spreads attention values across atoms in the AtomSpace.
- **MORK** provides an optimized storage and reduction backend for the AtomSpace.
