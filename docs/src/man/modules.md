# [Modules](@id modules-page)

This project implements a number of ART-based models with options that modulate their behavior (see the [options section of the Guide](@ref art_options))

This page lists both the [implemented models](@ref Implemented-Models) and some of their [variants](@ref variants)

## Implemented Models

This project has implementations of the following ART (unsupervised) and ARTMAP (supervised) modules:

```@meta
CurrentModule=AdaptiveResonance
```

- ART
  - [`FuzzyART`](@ref): Fuzzy ART
  - [`DVFA`](@ref): Dual Vigilance Fuzzy ART
  - [`DDVFA`](@ref): Distributed Dual Vigilance Fuzzy ART
- ARTMAP
  - [`SFAM`](@ref): Simplified Fuzzy ARTMAP
  - [`FAM`](@ref): Fuzzy ARTMAP
  - [`DAM`](@ref): Default ARTMAP

## Variants

Each module contains many [options](@ref art_options) that modulate its behavior.
Some of these options are used to modulate the internals of the module, such as switching the match and activation functions, to achieve different modules that are found in the literature.

These variants are:

- [`Gamma-Normalized FuzzyART`](@ref Gamma-Normalized-FuzzyART)

### Gamma-Normalized FuzzyART

A Gamma-Normalized FuzzyART is a FuzzyART module where the gamma normalization option is set on `gamma_normalization=true` and the kernel width parameter is set to $$\gamma >= 1.0$$ ($$\gamma_{ref}$$ is 1.0 by default):

```julia
my_gnfa = FuzzyART(gamma_normalization=true, gamma=5.0)
```

The `gamma_normalization` flag must be set high here because it also changes the thresholding value and match function of the module.
