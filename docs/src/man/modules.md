# [Modules](@id modules-page)

This project implements a number of ART-based models with options that modulate their behavior (see the [options section of the Guide](@ref art_options))

This page lists both the [implemented models](@ref Implemented-Models) and some of their [variants](@ref modules-variants)

Constructor and option details are grouped in the [ART reference](reference/models-art.md) and [ARTMAP reference](reference/models-artmap.md). Shared operations are documented under [Training](reference/training.md) and [Classification and evaluation](reference/classification.md).

## Implemented Models

This project has implementations of the following ART (unsupervised) and ARTMAP (supervised) modules:

```@meta
CurrentModule=AdaptiveResonance
```

- ART
  - [`FuzzyART`](@ref): Fuzzy ART
  - [`HypersphereART`](@ref): Hypersphere ART
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
  - [`MergeART`](@ref)
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

## Hypersphere ART

[`HypersphereART`](@ref) represents each category by a center and radius, stored
in a column of `art.W` as `[center; radius]`. It supports the same batch and
incremental `train!` and `classify` calls, optional training labels, `get_bmu`,
progress display, and epoch limit as the other ART models.

```julia
art = HypersphereART(rho=0.6, beta=1.0)
x = [0.0 0.1 0.9 1.0; 0.0 0.2 0.8 1.0]
labels = train!(art, x)
predictions = classify(art, x)
```

Raw inputs are normalized using `DataConfig`. Hypersphere ART does not use
complement coding: `preprocessed=true` means that the original features have
already been normalized. For incremental raw input, configure the feature
bounds first, for example with `art.config = DataConfig(0, 1, 2)`.

The options [`opts_HypersphereART`](@ref) include vigilance `rho`, choice `alpha`,
and learning rate `beta`. The radial extent `r_bar` defaults to `sqrt(dim) / 2`, the radius of the enclosing sphere
for the normalized unit cube. An explicit positive `r_bar` controls the distance
scale for category choice and vigilance; for custom input scales, choose it
at least as large as half the maximum pairwise sample distance. With `beta=1`,
a resonating sphere expands just enough to enclose an exterior sample while
retaining its previously enclosed points. Interior samples leave it unchanged.

Like FuzzyART, HypersphereART selects its formulas through the `activation`,
`match`, and `update` option symbols. Their defaults are
`:hypersphere_activation`, `:hypersphere_match`, and `:hypersphere_update`,
implemented in `src/lib/symbols.jl`. Replacement functions must support the
`[center; radius]` weight layout; fuzzy formulas assume a different layout.

## MergeART

Instead of being an ART module of its own, MergeART actually combines a trained DDVFA partition and then compresses prototypes inside each resulting cluster.
It accepts a DDVFA model rather than raw training samples:

```julia
source = DDVFA(rho_lb=0.7, rho_ub=0.85)
train!(source, X)
merged = MergeART(source; max_iter=10)
y_hat = classify(merged, X_test; get_bmu=true)
```

Note that the source `DDVFA` model is unchanged.
`merged.source_map[i]` is the new cluster corresponding to source F2 node `i`.
Calling `train!(merged, source)` again rebuilds from the current source snapshot; it does not import its counts a second time.
Source labels do not constrain this unsupervised procedure.
Merging parameters default to the source settings and can be overridden; compression uses reference exponent one, so `gamma` must exceed one.
Zero-norm prototypes use an explicit convention: a zero-norm input matches only a zero-norm destination.

MergeART supports all six DDVFA linkage methods and ordinary vector/batch inference.
It is excluded from `ART_MODULES`, which lists models trained on raw samples.
