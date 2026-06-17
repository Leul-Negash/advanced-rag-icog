# MeTTa (Meta Type Talk)

MeTTa, which stands for **Meta Type Talk**, is the primary programming language of
OpenCog Hyperon. It is a language for representing and transforming knowledge held
in the AtomSpace. MeTTa replaces the Atomese representation and Scheme scripting
that were used in OpenCog Classic.

## Key characteristics

MeTTa is a multi-paradigm language that combines:

- **Functional programming** - programs are expressions that are rewritten.
- **Logic / pattern matching** - rules match patterns of atoms and rewrite them.
- **A gradual, programmable type system** - types are themselves atoms and can be
  manipulated by the program.

The core execution model of MeTTa is **rewriting**: the interpreter repeatedly
matches expressions against equalities and rewrites them until no more rewrites
apply. This makes MeTTa well suited to symbolic AI and meta-programming, where a
program can reason about and modify its own rules.

## Implementations of MeTTa

There are several implementations of the MeTTa language:

- **Hyperon Experimental (hyperon / MeTTa-Rust)** - the reference implementation,
  written in Rust, available through the `hyperon` Python package.
- **MeTTaLog (metta-wam)** - an implementation that runs MeTTa on the Warren
  Abstract Machine (WAM) using SWI-Prolog. It targets compatibility with the
  reference implementation while adding compiler and debugging tooling.
- **PeTTa** - an efficient MeTTa implementation written in Prolog.

## Relationship to other technologies

- MeTTa operates on the **AtomSpace**.
- MeTTaLog and PeTTa are both implemented on top of **SWI-Prolog**.
- The **metta-moses** project is written in MeTTaLog.
- High-performance MeTTa execution can use **MORK** as a backend, and **FAISS**
  for vector-based atom spaces.

MeTTa is central to iCog Labs' engineering work: many of the company's projects
are written in or target the MeTTa language.
