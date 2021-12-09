---
title: 'AdaptiveResonance.jl: A Julia Implementation of Adaptive Resonance Theory (ART) Algorithms'
tags:
  - Julia
  - ART
  - Adaptive Resonance Theory
  - Machine Learning
  - Clustering
  - Neural Networks
authors:
  - name: Sasha Petrenko
    orcid: 0000-0003-2442-8901
    affiliation: 1
  - name: Donald C. Wunsch II
    orcid: 0000-0002-9726-9051
    affiliation: 1
affiliations:
  - name: Missouri University of Science and Technology
    index: 1
date: 1 June 2021
bibliography: paper.bib
---

# Summary

AdaptiveResonance.jl is a Julia package for machine learning with Adaptive Resonance Theory (ART) algorithms, written in the numerical computing language Julia.
ART is a neurocognitive theory of how competitive cellular networks can learn distributed patterns without supervision through recurrent field connections, eliciting the mechanisms of perception, expectation, and recognition [@Grossberg2013; @Grossberg1980].

# Statement of Need

There exist many variations of algorithms built upon ART [@DaSilva2019].
Each variation is related by utilizing recurrent connections of fields, driven by learning through match and mismatch of distributed patterns, and though they all differ in the details of their implementations, their algorithmic and programmatic requirements are often very similar.
Despite the relevance and successes of this class of algorithms in the literature, there does not exist to date a unified repository of their implementations in Julia.
The purpose of this package is to create a unified framework and repository of ART algorithms in Julia.

## Target Audience

This package is principally intended as a resource for researchers in machine learning and adaptive resonance theory for testing and developing new ART algorithms.
However, implementing these algorithms in the Julia language brings all of the benefits of the Julia itself, such as the speed of being implemented in a low-level language such as C while having the transparency of a high-level language such as MATLAB.
Being implemented in Julia allows the package to be understood and expanded upon by research scientists while still being able to be used in resource-demanding production environments.

# Adaptive Resonance Theory

ART is originally a theory of how competitive fields of neurons interact to form stable representations without supervision, and ART algorithms draw from this theory as biological inspiration for their design.
It is not strictly necessary to have an understanding of the theory to understand the use of the algorithms, but they share a common nomenclature that makes knowledge of the former useful for the latter.

## Theory

Adaptive resonance theory is a collection of neurological study from the neuron level to the network level [@ARTHestenes1987].
ART begins with a set of neural field differential equations, and the theory tackles problems from why sigmoidal activations are used and the conditions of stability for competitive neural networks [@Cohen1983a] to how the mammalian visual system works [@Grossberg2009] and the hard problem of consciousness linking resonant states to conscious experiences [@Grossberg2017].
Stephen Grossberg and Gail Carpenter have published many resources for learning the theory and its history in detail [@grossberg2021conscious].

## Algorithms

ART algorithms are generally characterized in behavior by the following:

1. They are inherently *unsupervised* learning algorithms at their core, but they have been adapted to supervised and reinforcement learning paradigms with frameworks such as ARTMAP [@Carpenter1991; @Carpenter1992] and FALCON [@Tan2019], respectively.
2. They are *incremental* learning algorithms, adjusting their weights or creating new ones at every sample presentation.
3. They are *neurogenesis* neural networks, representing their learning by the modification of existing prototype weights or instantiating new ones entirely.
4. They belong to the class of *competitive* neural networks, which compute their outputs with more complex dynamics than feedforward activation.

Because of the breadth of the original theory and variety of possible applications, ART-based algorithms are diverse in their implementation details.
Nevertheless, they are generally structured as follows:

1. ART models typically have two layers/fields denoted F1 and F2.
2. The F1 field is the feature representation field.
Most often, it is simply the input feature sample itself (after some necessary preprocessing).
3. The F2 field is the category representation field.
With some exceptions, each node in the F2 field represents its own category.
This is most easily understood as a weight vector representing a prototype for a class or centroid of a cluster.
4. An activation function is used to find the order of categories "most activated" for a given sample in F1.
5. In order of highest activation, a match function is used to compute the agreement between the sample and the categories.
6. If the match function for a category evaluates to a value above a threshold known as the vigilance parameter ($$\rho$$), the weights of that category may be updated according to a learning rule.
7. If there is complete mismatch across all categories, then a new categories is created according to some instantiation rule.

# Acknowledgements

This package is developed and maintained with sponsorship by the Applied Computational Intelligence Laboratory (ACIL) of the Missouri University of Science and Technology.
This project is supported by grants from the Army Research Labs Night Vision Electronic Sensors Directorate (NVESD), the DARPA Lifelong Learning Machines (L2M) program, Teledyne Technologies, and the National Science Foundation.
The material, findings, and conclusions here do not necessarily reflect the views of these entities.

<!-- This package is developed and maintained by [Sasha Petrenko](https://github.com/AP6YC) with sponsorship by the [Applied Computational Intelligence Laboratory (ACIL)](https://acil.mst.edu/). This project is supported by grants from the [Night Vision Electronic Sensors Directorate](https://c5isr.ccdc.army.mil/inside_c5isr_center/nvesd/), the [DARPA Lifelong Learning Machines (L2M) program](https://www.darpa.mil/program/lifelong-learning-machines), [Teledyne Technologies](http://www.teledyne.com/), and the [National Science Foundation](https://www.nsf.gov/).
The material, findings, and conclusions here do not necessarily reflect the views of these entities. -->

# References