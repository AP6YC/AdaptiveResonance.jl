```@meta
DocTestSetup = quote
    using AdaptiveResonance, Dates
end
```

![header](assets/header.png)

---

These pages serve as the official documentation for the AdaptiveResonance.jl Julia package.

Adaptive Resonance Theory (ART) began as a neurocognitive theory of how fields of cells can continuously learn stable representations, and it evolved into the basis for a myriad of practical machine learning algorithms.
Pioneered by Stephen Grossberg and Gail Carpenter, the field has had contributions across many years and from many disciplines, resulting in a plethora of engineering applications and theoretical advancements that have enabled ART-based algorithms to compete with many other modern learning and clustering algorithms.

The purpose of this package is to provide a home for the development and use of these ART-based machine learning algorithms.

See the [Index](@ref main-index) for the complete list of documented functions and types.

## Manual Outline

This documentation is split into the following sections:

```@contents
Pages = [
    "man/guide.md",
    "../examples/index.md",
    "man/modules.md",
    "man/contributing.md",
    "man/full-index.md",
    "man/dev-index.md",
]
Depth = 1
```

The [Package Guide](@ref) provides a tutorial to the full usage of the package, while [Examples](@ref examples) gives sample workflows using a variety of ART modules.
A list of the implemented ART modules is included in [Modules](@ref modules-page), where different options are also listed for creating variants of these modules that exist in the literature.

Instructions on how to contribute to the package are found in [Contributing](@ref), and docstrings for every element of the package is listed in the [Index](@ref main-index).
Names internal to the package are also listed under the [Developer Index](@ref dev-main-index).

## Documentation Build

This documentation was built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) with the following version and OS:

```@example
using AdaptiveResonance, Dates # hide
println("AdaptiveResonance v$(ADAPTIVERESONANCE_VERSION) docs built $(Dates.now()) with Julia $(VERSION) on $(Sys.KERNEL)") # hide
```

## Citation

If you make use of this project, please generate your citation with the [CITATION.cff](../../CITATION.cff) file of the repository.
Alternatively, you may use the following BibTeX entry for the JOSS paper associated with the repository:

```bibtex
@article{Petrenko2022,
  doi = {10.21105/joss.03671},
  url = {https://doi.org/10.21105/joss.03671},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {73},
  pages = {3671},
  author = {Sasha Petrenko and Donald C. Wunsch},
  title = {AdaptiveResonance.jl: A Julia Implementation of Adaptive Resonance Theory (ART) Algorithms},
  journal = {Journal of Open Source Software}
}
```
