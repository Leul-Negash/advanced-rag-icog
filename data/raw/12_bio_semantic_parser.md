# bio-semantic-parser

The bio-semantic-parser is a project developed at iCog Labs that converts
biomedical text into structured knowledge that can be stored in the AtomSpace. It
is part of iCog Labs' work on applying OpenCog Hyperon technologies to the life
sciences.

## Purpose

Biomedical literature contains a vast amount of knowledge expressed in natural
language - relationships between genes, proteins, diseases, and drugs. The
bio-semantic-parser reads this text and extracts entities and the relationships
between them, producing a structured representation suitable for symbolic
reasoning.

## How it fits the ecosystem

Once biomedical relationships are extracted, they can be represented as atoms
(nodes and links) in the AtomSpace. Reasoning components such as PLN can then draw
inferences over this biomedical knowledge - for example, suggesting possible
gene-disease associations that are not stated explicitly in any single document.

The bio-semantic-parser is an example of a domain-specific knowledge pipeline:
text in, structured knowledge graph out, reasoning on top. This mirrors the idea
of GraphRAG, where a knowledge graph derived from documents supports relationship-
based question answering.

## Coreference resolution

A standalone coreference-resolution service (built on the LingMess model) is used
to link mentions such as "it" or "the protein" back to the entity they refer to,
improving the quality of the extracted relationships before they are written to the
AtomSpace.

## Relationship to other components

- bio-semantic-parser **is developed by** iCog Labs.
- bio-semantic-parser **writes knowledge to** the AtomSpace.
- bio-semantic-parser **enables** PLN reasoning over biomedical knowledge.
