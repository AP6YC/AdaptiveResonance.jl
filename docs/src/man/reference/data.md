# Data Preparation

Configure feature bounds before incremental training. Normalization and complement coding requirements depend on the model; see the [model guide](@ref modules-page).

```@meta
CurrentModule = AdaptiveResonance
```

## Index

- [`DataConfig`](@ref), [`data_setup!`](@ref), [`get_data_characteristics`](@ref)
- [`linear_normalization`](@ref), [`complement_code`](@ref)

## Configuration

### [`DataConfig`](@id reference-DataConfig)

```@docs
DataConfig
```

### [`data_setup!`](@id reference-data_setup-mutating)

```@docs
data_setup!
```

### [`get_data_characteristics`](@id reference-get_data_characteristics)

```@docs
get_data_characteristics
```

## Transformations

### [`linear_normalization`](@id reference-linear_normalization)

```@docs
linear_normalization
```

### [`complement_code`](@id reference-complement_code)

```@docs
complement_code
```
