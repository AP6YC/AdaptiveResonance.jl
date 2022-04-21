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
Being implemented in Julia allows the package to be understood and expanded upon by research scientists while also being able to be used in resource-demanding production environments.

## Comparison to Existing Implementations

There exist a myriad of open implementations of ART algorithms that are the result of reproducibility efforts in the ART literature.
The Boston University Department of Cognitive and Neural Systems (CNS) Technology Laboratory software repository contains many one of the largest collections of algorithms and utilities related to ART, principally implemented in the MATLAB and C++ programming languages, from demonstrations of the learning laws of ART to implementations of ART and ARTMAP modules [@CNS_Software].
However, this repository serves as a codebase for the reproducibility of the software's associated CNS papers rather than as a single unified framework for ART implementations.

The Missouri University of Science and Technology Applied Computational Intelligence Laboratory (ACIL) hosts a myriad of individual ART algorithm implementations on its public GitHub group repository page [@ACIL_GitHub], chiefly implemented in the MATLAB and Python programming languages.
Though these ART algorithms are designed for open use, so too do they principally serve the reproducibility of their associated ACIL papers.

The ACIL group GitHub page additionally contains the NuART-Py library, which organizes a suite of clustering and biclustering ART algorithms as a distributable package in the Python language [@NuART-Py].
A similar package exists in the Java programming language in a separate repository containing only fundamental ART algorithms [@JavaART], and a new package in the R statistical programming language has only begun development at the time of this writing [@R_FuzzyART].

Though each of these ART software projects (and the very many and disparate implementations of individual algorithms in the literature) combined may implement the majority of ART algorithms relevant to modern research and engineering, together they lack cohesion in programming language and usage.
When considering ease of use and barrier to entry, many of these projects may be difficult to utilize for those less versed in the ART literature who might still significantly benefit from their use and understanding.

Lastly, very many ART implementations exist in the MATLAB programming language due to its popularity amongst the research scientists that have been the theory's primary clientele, which is at the detriment to those without private MATLAB licenses in research and industry.
The Julia programming language is selected for this open-source ART package implementation due to its syntactic ease of use and speed of development without comprimising computational efficiency due to the language's just-in-time compilation.

# Adaptive Resonance Theory

ART is originally a theory of how competitive fields of neurons interact to form stable representations without supervision, and ART algorithms draw from this theory as biological inspiration for their design.
It is not strictly necessary to have an understanding of the theory to understand the use of the algorithms, but they share a common nomenclature that makes knowledge of the former useful for the latter.

## Theory

Adaptive resonance theory is a collection of neurological study from the neuron level to the network level [@ARTHestenes1987].
ART begins with a set of neural field differential equations and procedurally tackles problems such as why sigmoidal neural activations are used, the conditions of stability for competitive neural networks [@Cohen1983a], how the mammalian visual system works [@Grossberg2009], and the hard problem of consciousness linking resonant states to conscious experiences [@Grossberg2017; @grossberg2021conscious].

## Algorithms

ART algorithms are generally characterized in behavior by the following:

1. They are inherently *unsupervised* learning algorithms at their core, but they have been adapted to supervised and reinforcement learning paradigms with frameworks such as ARTMAP [@Carpenter1991; @Carpenter1992] and FALCON [@Tan2019], respectively.
2. They are *incremental* learning algorithms, adjusting their weights or creating new ones at every sample presentation.
3. They are *neurogenesis* neural networks, representing their learning by the modification of existing prototype weights or instantiating new ones entirely.
4. They belong to the class of *competitive* neural networks, which compute their outputs with more complex dynamics than feedforward activation.

Because of the breadth of the original theory and variety of possible applications, ART-based algorithms are diverse in their nomenclature and implementation details.
Nevertheless, they are generally structured as follows:

1. ART models typically have two layers/fields denoted F1 and F2.
2. The F1 field is the feature representation field.
Most often, it is simply the input feature sample itself (after some requisite feature preprocessing, depending on the model).
3. The F2 field is the category representation field.
With some exceptions, each node in the F2 field represents its own category.
This is most easily interpreted as a weight vector representing a prototype for a class or centroid of a cluster.
4. An activation function is used to find the order of categories "most activated" for a given sample in F1.
5. In order of highest activation, a match function is used to compute the agreement between the sample and the categories.
6. If the match function for a category evaluates to a value above a threshold known as the vigilance parameter, the weights of that category may be updated according to a learning rule.
7. If there is complete mismatch across all categories, then a new category is created according to an instantiation rule determined by the algorithm in question.

# Acknowledgements

This package is developed and maintained with sponsorship by the Applied Computational Intelligence Laboratory (ACIL) of the Missouri University of Science and Technology.
This project is supported by grants from the Army Research Labs Night Vision Electronic Sensors Directorate (NVESD), the DARPA Lifelong Learning Machines (L2M) program, Teledyne Technologies, and the National Science Foundation.
The material, findings, and conclusions here do not necessarily reflect the views of these entities.

# References