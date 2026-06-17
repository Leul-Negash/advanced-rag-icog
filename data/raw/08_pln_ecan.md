# PLN and ECAN

OpenCog and OpenCog Hyperon include two cognitive components that operate over the
AtomSpace: PLN for reasoning and ECAN for attention allocation.

## PLN (Probabilistic Logic Networks)

PLN stands for **Probabilistic Logic Networks**. It is an uncertain-inference
framework that combines probability theory with term and predicate logic. PLN
reasons over the **truth values** attached to atoms in the AtomSpace, allowing the
system to draw conclusions even when knowledge is incomplete or uncertain.

PLN supports operations such as deduction, induction, and abduction on uncertain
knowledge. It was developed within the OpenCog project and is closely associated
with Ben Goertzel, who co-authored the foundational book on Probabilistic Logic
Networks.

## ECAN (Economic Attention Networks)

ECAN stands for **Economic Attention Networks**. It is the attention-allocation
mechanism of OpenCog. ECAN assigns **attention values** to atoms - typically
Short-Term Importance (STI) and Long-Term Importance (LTI) - and spreads these
values across the AtomSpace using a model inspired by economics, where attention
is a limited currency that flows to the most useful atoms.

ECAN decides which atoms the system should focus its limited computational
resources on, forming an "attentional focus" of the most currently relevant
knowledge.

## Cognitive synergy

PLN and ECAN are designed to work together with the other components. ECAN focuses
attention on a relevant subset of the AtomSpace, and PLN then reasons over that
subset. MOSES can learn new programs, and MeTTa ties everything together. This
mutual cooperation is the principle of **cognitive synergy** that underlies the
OpenCog and Hyperon architectures.

## Relationship to other components

- PLN **reasons over** truth values of atoms in the AtomSpace.
- ECAN **allocates attention** across atoms in the AtomSpace.
- PLN and ECAN **support** the cognitive-synergy goal of OpenCog Hyperon.
