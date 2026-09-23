# Training and Category Lifecycle

Internal operations used to initialize, search, and update models.
Each function includes its documented dispatch variants.

```@meta
CurrentModule = AdaptiveResonance
```

## Index

- [`initialize!`](@ref), [`set_threshold!`](@ref), [`stopping_conditions`](@ref)
- [`create_category!`](@ref), [`activation_match!`](@ref), [`learn!`](@ref)
- [`resonance_search!`](@ref), [`resonance_activation!`](@ref), [`resonance_match!`](@ref), [`resonance_match_scale`](@ref)
- [`merge_categories!`](@ref), [`compress_categories!`](@ref), [`prototype_similarity`](@ref)
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

## Resonance Search

All implemented ART and ARTMAP learners share `resonance_search!` for candidate
traversal, vigilance checks, optional supervisory match tracking, mismatch
fallback, and statistics. Learning and category creation remain with the caller.
The hooks below specialize activation preparation, lazy match evaluation, and
conversion of the tracking increment to each module's match units.

### [`resonance_search!`](@id internals-resonance_search-mutating)

```@docs
resonance_search!
```

### [`resonance_activation!`](@id internals-resonance_activation-mutating)

```@docs
resonance_activation!
```

### [`resonance_match!`](@id internals-resonance_match-mutating)

```@docs
resonance_match!
```

### [`resonance_match_scale`](@id internals-resonance_match_scale)

```@docs
resonance_match_scale
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

## Cluster Merging and Compression

```@docs
merge_categories!
compress_categories!
prototype_similarity
```
