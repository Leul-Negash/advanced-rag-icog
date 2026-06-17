# PeTTa

PeTTa is an efficient implementation of the **MeTTa** language written in
**Prolog**. It is maintained primarily by the developer known as patham9 and is
distributed under the `trueagi-io` organization. PeTTa lets developers run MeTTa
programs using SWI-Prolog as the underlying engine.

## Dependencies

PeTTa requires:

- **SWI-Prolog** version 9.3.x or higher.
- **Python 3.x**, used for the `janus` Python interoperability layer.

## MORK and FAISS spaces

If **MORK** and **FAISS** are installed, PeTTa can build support for:

- **MORK-based atom spaces** - high-performance metagraph storage, via the
  `mork_ffi` bindings (which depend on `trueagi-io/mork`).
- **FAISS-based atom-vector spaces** - vector/embedding search over atoms, via the
  `faiss_ffi` bindings (which depend on `facebookresearch/faiss`).

These are enabled by running PeTTa's `build.sh` script, which clones and builds the
`mork_ffi` and `faiss_ffi` repositories.

## Tooling

PeTTa supports interactive development:

- A **Jupyter kernel** (`jupyter-petta-kernel`) allows MeTTa development inside
  notebooks.
- A **MeTTa HTTP server** (MettaWamJam) can run MeTTa code over HTTP.
- A set of extension libraries can be invoked directly from MeTTa files.

## Relationship to other components

- PeTTa **implements** the MeTTa language.
- PeTTa **runs on** SWI-Prolog.
- PeTTa **uses** MORK (via `mork_ffi`) and **FAISS** (via `faiss_ffi`) for
  optional high-performance atom spaces.
