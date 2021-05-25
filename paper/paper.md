---
title: 'AdaptiveResonance: A Julia implementation of Adaptive Resonance Theory (ART) algorithms'
tags:
  - Julia
  - ART
  - Adaptive Resonance Theory
  - Machine Learning
  - Clustering
  - Neural Networks
authors:
  - name: Sasha Petrenko^[Missouri University of Science and Technology]
    orcid: 0000-0003-2442-8901
date: 25 May 2021
bibliography: paper.bib
---

# Summary

AdaptiveResonance is a library for machine learning with Adaptive Resonance Theory (ART) algorithms, written in the numerical computing language Julia.
ART is a neurocognitive theory of how competitive cellular networks can learn distributed patterns without supervision through recurrent field connections, eliciting the mechanisms of perception, expectation, and recognition [@Grossberg2013; @Grossberg1980].

# Statement of need

There exist many variations of algorithms built upon ART [@DaSilva2019].
Each variation is related by utilizing recurrent connections of fields, driven by learning through match and mismatch of distributed patterns, and though they all differ in the details of their implementations, their algorithmic and programmatic requirements are often very similar.
Despite the relevance and successes of this class of algorithms in the literature, there does not exist to date a unified repository of their implementations in Julia.
The purpose of this package is to create a unified framework and repository of ART algorithms in Julia.

# Acknowledgements

This package is developed and maintained with sponsorship by the Applied Computational Intelligence Laboratory (ACIL) of the Missouri University of Science and Technology.
This project is supported by grants from the Army Research Labs Night Vision Electronic Sensors Directorate (NVESD), the DARPA Lifelong Learning Machines (L2M) program, Teledyne Technologies, and the National Science Foundation.
The material, findings, and conclusions here do not necessarily reflect the views of these entities.

<!-- This package is developed and maintained by [Sasha Petrenko](https://github.com/AP6YC) with sponsorship by the [Applied Computational Intelligence Laboratory (ACIL)](https://acil.mst.edu/). This project is supported by grants from the [Night Vision Electronic Sensors Directorate](https://c5isr.ccdc.army.mil/inside_c5isr_center/nvesd/), the [DARPA Lifelong Learning Machines (L2M) program](https://www.darpa.mil/program/lifelong-learning-machines), [Teledyne Technologies](http://www.teledyne.com/), and the [National Science Foundation](https://www.nsf.gov/).
The material, findings, and conclusions here do not necessarily reflect the views of these entities. -->

# References