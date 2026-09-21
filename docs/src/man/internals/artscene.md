# ARTSCENE Internals

Individual stages of the image filtering pipeline exposed through [`artscene_filter`](@ref).

```@meta
CurrentModule = AdaptiveResonance
```

## Index

- [`color_to_gray`](@ref), [`contrast_normalization`](@ref), [`surround_kernel`](@ref), [`ddt_x`](@ref)
- [`oriented_kernel`](@ref), [`ddt_y`](@ref), [`contrast_sensitive_oriented_filtering`](@ref), [`contrast_insensitive_oriented_filtering`](@ref)
- [`competition_kernel`](@ref), [`ddt_z`](@ref), [`orientation_competition`](@ref), [`patch_orientation_color`](@ref)

## Image Preprocessing

### [`color_to_gray`](@id internals-color_to_gray)

```@docs
color_to_gray
```

### [`contrast_normalization`](@id internals-contrast_normalization)

```@docs
contrast_normalization
```

### [`surround_kernel`](@id internals-surround_kernel)

```@docs
surround_kernel
```

### [`ddt_x`](@id internals-ddt_x)

```@docs
ddt_x
```

## Orientation Filtering

### [`oriented_kernel`](@id internals-oriented_kernel)

```@docs
oriented_kernel
```

### [`ddt_y`](@id internals-ddt_y)

```@docs
ddt_y
```

### [`contrast_sensitive_oriented_filtering`](@id internals-contrast_sensitive_oriented_filtering)

```@docs
contrast_sensitive_oriented_filtering
```

### [`contrast_insensitive_oriented_filtering`](@id internals-contrast_insensitive_oriented_filtering)

```@docs
contrast_insensitive_oriented_filtering
```

## Competition and Patches

### [`competition_kernel`](@id internals-competition_kernel)

```@docs
competition_kernel
```

### [`ddt_z`](@id internals-ddt_z)

```@docs
ddt_z
```

### [`orientation_competition`](@id internals-orientation_competition)

```@docs
orientation_competition
```

### [`patch_orientation_color`](@id internals-patch_orientation_color)

```@docs
patch_orientation_color
```
