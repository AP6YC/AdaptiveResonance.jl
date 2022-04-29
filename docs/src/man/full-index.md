# [Index](@id main-index)

This page lists the core methods and types of the `AdaptiveResonance.jl` package.
The [Methods](@ref index-methods) section lists the public methods for the package that use the modules in [Types](@ref index-types).
Each of these entries link to the docstrings in the [Docs](@ref index-docs) section.

ART modules document their internal working parameters and references, while their hyperparameters/options are documented under their corresponding option structs `opts_...`.

## [Methods](@id index-methods)

```@index
Modules = [AdaptiveResonance]
Order = [:function]
Public = true
```

## [Types](@id index-types)

```@index
Modules = [AdaptiveResonance]
Order = [:type]
Public = true
```

## [Docs](@id index-docs)

```@docs
AdaptiveResonance
train!
classify
data_setup!
performance
complement_code
get_data_characteristics
linear_normalization
get_data_shape
get_n_samples
DDVFA
DVFA
FuzzyART
DAM
FAM
SFAM
opts_DDVFA
opts_DVFA
opts_FuzzyART
opts_DAM
opts_FAM
opts_SFAM
DataConfig
ARTModule
ART
ARTMAP
ARTOpts
```
