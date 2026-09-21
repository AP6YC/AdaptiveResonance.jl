# [API Overview](@id main-index)

Browse the public API by task. Each name links to one reference location containing all of its documented methods. For help choosing an algorithm, start with the [model guide](@ref modules-page).

```@meta
CurrentModule = AdaptiveResonance
```

## [ART Models and Options](reference/models-art.md)

- [`FuzzyART`](@ref), [`opts_FuzzyART`](@ref)
- [`HypersphereART`](@ref), [`opts_HypersphereART`](@ref)
- [`DVFA`](@ref), [`opts_DVFA`](@ref)
- [`DDVFA`](@ref), [`opts_DDVFA`](@ref)
- [`GammaNormalizedFuzzyART`](@ref), [`opts_GammaNormalizedFuzzyART`](@ref)

## [ARTMAP Models and Options](reference/models-artmap.md)

- [`SFAM`](@ref), [`opts_SFAM`](@ref)
- [`FAM`](@ref), [`opts_FAM`](@ref)
- [`DAM`](@ref), [`opts_DAM`](@ref)

## [Training](reference/training.md)

- [`train!`](@ref)

## [Classification and Evaluation](reference/classification.md)

- [`classify`](@ref)
- [`performance`](@ref)

## [Data Preparation](reference/data.md)

- [`DataConfig`](@ref), [`data_setup!`](@ref), [`get_data_characteristics`](@ref)
- [`linear_normalization`](@ref), [`complement_code`](@ref)

## [Advanced Configuration](reference/configuration.md)

- [`ACTIVATION_FUNCTIONS`](@ref), [`MATCH_FUNCTIONS`](@ref), [`UPDATE_FUNCTIONS`](@ref)
- [`DDVFA_METHODS`](@ref), [`get_W`](@ref)
- [`ART_MODULES`](@ref), [`ARTMAP_MODULES`](@ref), [`ADAPTIVERESONANCE_MODULES`](@ref), [`ADAPTIVERESONANCE_VERSION`](@ref)
- [`ARTModule`](@ref), [`ART`](@ref), [`ARTMAP`](@ref), [`ARTOpts`](@ref)

## [ARTSCENE Feature Extraction](reference/artscene.md)

- [`artscene_filter`](@ref)

## Package

```@docs
AdaptiveResonance
```
