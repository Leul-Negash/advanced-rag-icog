# OpenCog Hyperon

OpenCog Hyperon is an open-source software framework for Artificial General
Intelligence (AGI). It is the successor to the original OpenCog framework, often
called "OpenCog Classic." Hyperon is designed and led by Ben Goertzel and is
developed primarily under the SingularityNET and TrueAGI organizations, with
significant engineering contributions from iCog Labs.

## Goals

Hyperon aims to provide a scalable, integrative platform that can host many
different AI paradigms - symbolic reasoning, probabilistic logic, evolutionary
program learning, and neural networks - inside a single cognitive architecture.
The guiding philosophy is "cognitive synergy": different reasoning methods help
each other overcome their individual bottlenecks.

## Core components

OpenCog Hyperon is built from several core components:

- **AtomSpace** - a metagraph knowledge store that holds all knowledge as atoms.
- **MeTTa** - the Meta Type Talk programming language used to represent and manipulate knowledge in the AtomSpace.
- **PLN (Probabilistic Logic Networks)** - an uncertain reasoning system.
- **ECAN (Economic Attention Networks)** - an attention-allocation mechanism.
- **MOSES** - an evolutionary program-learning component.

In Hyperon, MeTTa replaces the older Atomese language and Scheme bindings that
were used in OpenCog Classic. The AtomSpace remains the central knowledge store,
but Hyperon reimplements it as a distributed, high-performance metagraph.

## Relationship between Hyperon and OpenCog Classic

OpenCog Classic used Atomese and a C++ AtomSpace. OpenCog Hyperon is a ground-up
redesign that uses MeTTa as its primary language and introduces new high-
performance backends such as MORK for metagraph storage and reduction. Hyperon
is intended to scale to much larger knowledge bases than OpenCog Classic.

## Who builds Hyperon

Hyperon is a community effort. SingularityNET funds and coordinates much of the
work, Ben Goertzel provides the scientific direction, and iCog Labs contributes a
large share of the engineering, including language implementations and ecosystem
tooling. The code is hosted mainly under the `trueagi-io` GitHub organization.
