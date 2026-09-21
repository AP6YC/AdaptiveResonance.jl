# Training and Category Lifecycle

Internal operations used to initialize, search, and update models.
Each function includes its documented dispatch variants.

```@meta
CurrentModule = AdaptiveResonance
```

## Index

- [`initialize!`](@ref), [`set_threshold!`](@ref), [`stopping_conditions`](@ref)
- [`create_category!`](@ref), [`activation_match!`](@ref), [`learn!`](@ref), [`hypersphere_search`](@ref)
- [`get_n_weights`](@ref), [`get_n_weights_vec`](@ref)

## Initialization and Stopping

### [`initialize!`](@id internals-initialize-mutating)

```@docs
initialize!
```

### [`set_threshold!`](@id internals-set_threshold-mutating)

```@docs
set_threshold!
```

### [`stopping_conditions`](@id internals-stopping_conditions)

```@docs
stopping_conditions
```

## Category Creation and Learning

### [`create_category!`](@id internals-create_category-mutating)

```@docs
create_category!
```

### [`activation_match!`](@id internals-activation_match-mutating)

```@docs
activation_match!
```

### [`learn!`](@id internals-learn-mutating)

```@docs
learn!
```

### [`hypersphere_search`](@id internals-hypersphere_search)

```@docs
hypersphere_search
```

## Distributed Model State

### [`get_n_weights`](@id internals-get_n_weights)

```@docs
get_n_weights
```

### [`get_n_weights_vec`](@id internals-get_n_weights_vec)

```@docs
get_n_weights_vec
```
