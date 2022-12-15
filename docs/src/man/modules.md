# [Modules](@id modules-page)

This project implements a number of ART-based models with options that modulate their behavior (see the [options section of the Guide](@ref art_options))

This page lists both the [implemented models](@ref Implemented-Models) and some of their [variants](@ref modules-variants)

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

## [Variants](@id modules-variants)

Each module contains many [options](@ref art_options) that modulate its behavior.
Some of these options are used to modulate the internals of the module, such as switching the match and activation functions, to achieve different modules that are found in the literature.

These variants are:

- ART
  - [`GammaNormalizedFuzzyART`](@ref): Gamma-Normalized FuzzyART
- ARTMAP
  - [`DAM`](@ref): Default ARTMAP

### Gamma-Normalized FuzzyART

A [`Gamma-Normalized FuzzyART`](@ref GammaNormalizedFuzzyART) is implemented as a [`FuzzyART`](@ref) module where the gamma normalization option is set on `gamma_normalization=true` and the kernel width parameter is set to $$\gamma >= 1.0$$ ($$\gamma_{ref}$$ is 1.0 by default).
It can be created with the convenience constructor:

```julia
my_gnfa = GammaNormalizedFuzzyART()
```

Under the hood, this simply does

```julia
my_gnfa = FuzzyART(gamma_normalization=true)
```

which also sets the match and activation function options to `match=:gamma_match` and `activation=:gamma_activation`, respectively.

### Default ARTMAP

A [`Default ARTMAP`](@ref DAM) is implemented as a [`Simplified FuzzyARTMAP`](@ref SFAM) module where the activation function is set to Default ARTMAP's choice-by difference function via `activation=:choice_by_difference`.
It can be created with the convenience constructor:

```julia
my_dam = DAM()
```

Under the hood, this simply does

```julia
my_dam = SFAM(activation=:choice_by_difference)
```
