# MeTTaLog (metta-wam)

MeTTaLog is an implementation of the **MeTTa** language designed to run on the
**Warren Abstract Machine (WAM)**. Its repository is named **metta-wam**. MeTTaLog
uses **SWI-Prolog** as its underlying engine and aims for compatibility with the
reference Hyperon implementation while adding a compiler, interpreter, language
server (LSP), and extensive examples and tooling.

## Installation components

MeTTaLog's `INSTALL.sh` script sets up several components:

- **SWI-Prolog** version 9.3.9 or higher.
- **janus** - a Python package that interfaces with SWI-Prolog.
- **pyswip** - another Python-Prolog integration package.
- **hyperon** - the Hyperon Python package, used for running compatibility tests.
- **mettalog-vspace** - lets Rust-based MeTTa use extra functionality found in
  MeTTaLog (including MORK-backed spaces).
- **mettalog-jupyter-kernel** - work with `.metta` files in Jupyter notebooks.

## Compatibility testing

MeTTaLog runs a large test suite to measure compatibility with the reference
`hyperon` implementation. Continuous and nightly reports track which tests pass.

## Relationship to other components

- MeTTaLog **implements** the MeTTa language.
- MeTTaLog **runs on** the Warren Abstract Machine via SWI-Prolog.
- MeTTaLog **uses** the `hyperon` package for compatibility tests.
- The **metta-moses** project **is written in** MeTTaLog.
- MeTTaLog can **use** MORK through `mettalog-vspace`.
