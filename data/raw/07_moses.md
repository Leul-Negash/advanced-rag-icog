# MOSES and metta-moses

MOSES stands for **Meta-Optimizing Semantic Evolutionary Search**. It is a
machine-learning tool - an *evolutionary program learner* - that is capable of
learning short programs that capture patterns in input datasets. For a given data
input, the learned programs will roughly recreate the dataset on which they were
trained.

## Origins

MOSES is derived from the ideas formulated in **Moshe Looks'** 2006 PhD thesis,
"Competent Program Evolution" (Washington University in St. Louis, Missouri).
Moshe Looks is also one of the primary authors of the original MOSES code. MOSES
is part of the OpenCog project and is maintained under the `singnet` organization.

## How MOSES works

The term "evolutionary" means MOSES uses **genetic programming** techniques to
evolve new programs. Each program can be thought of as a tree, similar to a
decision tree, but allowing intermediate nodes to be any programming-language
construct. Evolution proceeds by selecting one exemplar tree from a collection of
reasonably fit individuals, and then making random alterations to the program tree
to try to find a fitter (more accurate) program. Programs can be output in the
`combo` programming language or in Python.

## Applications

MOSES has been used in several commercial applications, including:

- The analysis of medical patient and physician clinical data.
- Several different financial systems.

It is also used by OpenCog to learn automated behaviors, movements, and actions in
response to perceptual stimuli of artificial-life virtual agents (for example,
pet-dog game avatars). Future plans include learning behavioral programs for
real-world robots via the OpenPsi implementation of Psi-theory and ROS nodes
running on the OpenCog AtomSpace.

## metta-moses

**metta-moses** is a reimplementation of the MOSES algorithm in **MeTTaLog**. The
original algorithm comes from the `asmoses` repository in the OpenCog project, and
metta-moses ports it to the MeTTa ecosystem. It is developed under the
`iCog-Labs-Dev` GitHub organization and requires MeTTaLog (SWI-Prolog 9.3.x) to
run.

## Relationship to other components

- metta-moses **reimplements** MOSES.
- metta-moses **is written in** MeTTaLog.
- MOSES **is used by** OpenCog for program learning.
- MOSES **runs on** the AtomSpace via OpenPsi for robot control.
