# Modules

This project implements a number of ART-based models with options that modulate their behavior (see the [options section of the Guide](@ref art_options))

This page lists both the [implemented models](@ref Implemented-Models) and some [variants](@ref Variants)

## Implemented Models

This project has implementations of the following ART (unsupervised) and ARTMAP (supervised) modules:

- ART
  - `FuzzyART`: Fuzzy ART
  - `DVFA`: Dual Vigilance Fuzzy ART
  - `DDVFA`: Distributed Dual Vigilance Fuzzy ART
- ARTMAP
  - `SFAM`: Simplified Fuzzy ARTMAP
  - `FAM`: Fuzzy ARTMAP
  - `DAM`: Default ARTMAP

## Variants

Each module contains many [options](@ref art_options) that modulate its behavior.
Some of these options are used to modulate the internals of the module, such as switching the match and activation functions, to achieve different modules that are found in the literature.

These variants are:

- [`Gamma-Normalized FuzzyART`](@ref Gamma-Normalized-FuzzyART)

### Gamma-Normalized FuzzyART

A Gamma-Normalized FuzzyART is a FuzzyART module where the kernel width parameter is set to $$\gamma = \gamma_{ref} = 1.0$$ ($$\gamma_{ref}$$ is 1.0 by default):

```julia
my_gnfa = FuzzyART(gamma=1)
```
