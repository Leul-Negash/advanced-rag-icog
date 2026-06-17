# MORK (MeTTa Optimal Reduction Kernel)

MORK stands for the **MeTTa Optimal Reduction Kernel**. It is a high-performance
backend for storing and reducing metagraphs in the OpenCog Hyperon ecosystem.
MORK is developed under the `trueagi-io` organization and is intended to give
MeTTa and the AtomSpace fast, scalable storage and pattern matching.

## Purpose

The AtomSpace can grow to billions of atoms. To make reasoning practical at this
scale, MORK provides:

- A compact in-memory and on-disk representation of the metagraph.
- Extremely fast pattern matching and query over atoms.
- A reduction engine that rewrites MeTTa expressions efficiently.

MORK is designed so that MeTTa programs can run against very large knowledge bases
without the AtomSpace becoming a bottleneck.

## How MORK is used

Other projects integrate MORK as a backend:

- **PeTTa** can use MORK-based atom spaces (and FAISS-based atom-vector spaces)
  when MORK and FAISS are installed. PeTTa relies on the `mork_ffi` bindings,
  which depend on `trueagi-io/mork`.
- **MeTTaLog** can use MORK for extra functionality through the
  `mettalog-vspace` component.

## Relationship to other components

- MORK is a backend for the **AtomSpace**.
- MORK accelerates the execution of **MeTTa** programs.
- **PeTTa** uses MORK through the `mork_ffi` foreign-function interface.
- MORK is complemented by **FAISS** when vector (embedding) search over atoms is
  required.
