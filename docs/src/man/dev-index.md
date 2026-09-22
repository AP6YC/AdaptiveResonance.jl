# [Developer Reference](@id dev-main-index)

This page lists the types and methods that are internal to the `AdaptiveResonance.jl` package.
Because these are internal types, they are organized here by responsibility.
These internal names may change between versions and are not part of the public API.

```@meta
CurrentModule = AdaptiveResonance
```

## [Type Hierarchy and State](internals/types.md)

- [`SingleART`](@ref), [`AbstractFuzzyART`](@ref), [`MergeART`](@ref), [`opts_MergeART`](@ref)
- [`ARTMatrix`](@ref), [`ARTVector`](@ref), [`ARTStats`](@ref), [`ARTIterator`](@ref), [`ART_DIM`](@ref), [`ART_SAMPLES`](@ref)

## [Training and Category Lifecycle](internals/lifecycle.md)

- [`initialize!`](@ref), [`set_threshold!`](@ref), [`stopping_conditions`](@ref)
- [`create_category!`](@ref), [`activation_match!`](@ref), [`learn!`](@ref)
- [`resonance_search!`](@ref), [`resonance_activation!`](@ref), [`resonance_match!`](@ref), [`resonance_match_scale`](@ref)
- [`get_n_weights`](@ref), [`get_n_weights_vec`](@ref)

## [Activation, Matching, and Weight Updates](internals/activation.md)

- [`art_activation`](@ref), [`art_match`](@ref), [`art_learn`](@ref)
- [`basic_activation`](@ref), [`basic_match`](@ref), [`unnormalized_match`](@ref), [`basic_update`](@ref), [`gamma_activation`](@ref), [`gamma_match`](@ref), [`gamma_match_sub`](@ref), [`choice_by_difference`](@ref), [`x_W_min_norm`](@ref), [`W_norm`](@ref)
- [`hypersphere_activation`](@ref), [`hypersphere_match`](@ref), [`hypersphere_update`](@ref)
- [`similarity`](@ref), [`single`](@ref), [`average`](@ref), [`complete`](@ref), [`median`](@ref), [`weighted`](@ref), [`centroid`](@ref)

## [Data, Iteration, and Logging Utilities](internals/utilities.md)

- [`init_train!`](@ref), [`init_classify!`](@ref), [`hypersphere_preprocess`](@ref), [`get_dim`](@ref), [`get_n_samples`](@ref), [`get_data_shape`](@ref), [`get_sample`](@ref), [`element_min`](@ref)
- [`get_iterator`](@ref), [`update_iter`](@ref), [`build_art_stats`](@ref), [`log_art_stats!`](@ref)
- [`replace_mat_index!`](@ref), [`unsafe_replace_mat_index!`](@ref), [`accommodate_vector!`](@ref)
- [`_COMMON_DOC`](@ref), [`_OPTS_DOCSTRING`](@ref), [`_ARG_ART`](@ref), [`_ARG_X`](@ref), [`_ARG_W`](@ref), [`_ARG_INDEX`](@ref), [`_ARG_ART_X_W`](@ref), [`_ARGS_MATRIX_REPLACE`](@ref), [`MATCH_FUNCTIONS_DOCS`](@ref), [`ACTIVATION_FUNCTIONS_DOCS`](@ref)

## [ARTSCENE Internals](internals/artscene.md)

- [`color_to_gray`](@ref), [`contrast_normalization`](@ref), [`surround_kernel`](@ref), [`ddt_x`](@ref)
- [`oriented_kernel`](@ref), [`ddt_y`](@ref), [`contrast_sensitive_oriented_filtering`](@ref), [`contrast_insensitive_oriented_filtering`](@ref)
- [`competition_kernel`](@ref), [`ddt_z`](@ref), [`orientation_competition`](@ref), [`patch_orientation_color`](@ref)
