# Data, Iteration, and Logging Utilities

Internal helpers for preprocessing, iteration, storage, and documentation.

```@meta
CurrentModule = AdaptiveResonance
```

## Index

- [`init_train!`](@ref), [`init_classify!`](@ref), [`hypersphere_preprocess`](@ref), [`get_dim`](@ref), [`get_n_samples`](@ref), [`get_data_shape`](@ref), [`get_sample`](@ref), [`element_min`](@ref)
- [`get_iterator`](@ref), [`update_iter`](@ref), [`build_art_stats`](@ref), [`log_art_stats!`](@ref)
- [`replace_mat_index!`](@ref), [`unsafe_replace_mat_index!`](@ref), [`accommodate_vector!`](@ref)
- [`_COMMON_DOC`](@ref), [`_OPTS_DOCSTRING`](@ref), [`_ARG_ART`](@ref), [`_ARG_X`](@ref), [`_ARG_W`](@ref), [`_ARG_INDEX`](@ref), [`_ARG_ART_X_W`](@ref), [`_ARGS_MATRIX_REPLACE`](@ref), [`MATCH_FUNCTIONS_DOCS`](@ref), [`ACTIVATION_FUNCTIONS_DOCS`](@ref)

## Data Preparation

### [`init_train!`](@id internals-init_train-mutating)

```@docs
init_train!
```

### [`init_classify!`](@id internals-init_classify-mutating)

```@docs
init_classify!
```

### [`hypersphere_preprocess`](@id internals-hypersphere_preprocess)

```@docs
hypersphere_preprocess
```

### [`get_dim`](@id internals-get_dim)

```@docs
get_dim
```

### [`get_n_samples`](@id internals-get_n_samples)

```@docs
get_n_samples
```

### [`get_data_shape`](@id internals-get_data_shape)

```@docs
get_data_shape
```

### [`get_sample`](@id internals-get_sample)

```@docs
get_sample
```

### [`element_min`](@id internals-element_min)

```@docs
element_min
```

## Iteration and Logging

### [`get_iterator`](@id internals-get_iterator)

```@docs
get_iterator
```

### [`update_iter`](@id internals-update_iter)

```@docs
update_iter
```

### [`build_art_stats`](@id internals-build_art_stats)

```@docs
build_art_stats
```

### [`log_art_stats!`](@id internals-log_art_stats-mutating)

```@docs
log_art_stats!
```

## Storage Operations

### [`replace_mat_index!`](@id internals-replace_mat_index-mutating)

```@docs
replace_mat_index!
```

### [`unsafe_replace_mat_index!`](@id internals-unsafe_replace_mat_index-mutating)

```@docs
unsafe_replace_mat_index!
```

### [`accommodate_vector!`](@id internals-accommodate_vector-mutating)

```@docs
accommodate_vector!
```

## Docstring Templates

### [`_COMMON_DOC`](@id internals-_COMMON_DOC)

```@docs
_COMMON_DOC
```

### [`_OPTS_DOCSTRING`](@id internals-_OPTS_DOCSTRING)

```@docs
_OPTS_DOCSTRING
```

### [`_ARG_ART`](@id internals-_ARG_ART)

```@docs
_ARG_ART
```

### [`_ARG_X`](@id internals-_ARG_X)

```@docs
_ARG_X
```

### [`_ARG_W`](@id internals-_ARG_W)

```@docs
_ARG_W
```

### [`_ARG_INDEX`](@id internals-_ARG_INDEX)

```@docs
_ARG_INDEX
```

### [`_ARG_ART_X_W`](@id internals-_ARG_ART_X_W)

```@docs
_ARG_ART_X_W
```

### [`_ARGS_MATRIX_REPLACE`](@id internals-_ARGS_MATRIX_REPLACE)

```@docs
_ARGS_MATRIX_REPLACE
```

### [`MATCH_FUNCTIONS_DOCS`](@id internals-MATCH_FUNCTIONS_DOCS)

```@docs
MATCH_FUNCTIONS_DOCS
```

### [`ACTIVATION_FUNCTIONS_DOCS`](@id internals-ACTIVATION_FUNCTIONS_DOCS)

```@docs
ACTIVATION_FUNCTIONS_DOCS
```
