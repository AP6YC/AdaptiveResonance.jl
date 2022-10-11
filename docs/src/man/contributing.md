# Contributing

This page serves as the contribution guide for the `AdaptiveResonance.jl` package.
From top to bottom, the ways of contributing are:

- [GitHub Issues:](@ref Issues) how to raise an issue with the project.
- [Julia Development:](@ref Julia-Development) how to download and interact with the package.
- [GitFlow:](@ref GitFlow) how to directly contribute code to the package in an organized way on GitHub.
- [Development Details:](@ref Development-Details) how the internals of the package are currently setup if you would like to directly contribute code.

## Issues

The main point of contact is the [GitHub issues](https://github.com/AP6YC/AdaptiveResonance.jl/issues) page for the project.
This is the easiest way to contribute to the project, as any issue you find or request you have will be addressed there by the authors of the package.
Depending on the issue, the authors will collaborate with you, and after making changes they will link a [pull request](@ref GitFlow) which addresses your concern or implements your proposed changes.

## Julia Development

As a Julia package, development follows the usual procedure:

1. Clone the project from GitHub
2. Switch to or create the branch that you wish work on (see [GitFlow](@ref)).
3. Start Julia at your development folder.
4. Instantiate the package (i.e., download and install the package dependencies).

For example, you can get the package and startup Julia with

```sh
git clone git@github.com:AP6YC/AdaptiveResonance.jl.git
julia --project=.
```

!!! note "Note"
    In Julia, you must activate your project in the current REPL to point to the location/scope of installed packages.
    The above immediately activates the project when starting up Julia, but you may also separately startup the julia and activate the package with the interactive
    package manager via the `]` syntax:

    ```julia-repl
    julia
    julia> ]
    (@v1.8) pkg> activate .
    (AdaptiveResonance) pkg>
    ```

You may run the package's unit tests after the above setup in Julia with

```julia-repl
julia> using Pkg
julia> Pkg.instantiate()
julia> Pkg.test()
```

or interactively though the Julia package manager with

```julia-repl
julia> ]
(AdaptiveResonance) pkg> instantiate
(AdaptiveResonance) pkg> test
```

## GitFlow

As of verson `0.3.7`, the `AdaptiveResonance.jl` package follows the [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) git working model.
The [original post](https://nvie.com/posts/a-successful-git-branching-model/) by Vincent Driessen outlines this methodology quite well, while [Atlassian](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) has a good tutorial as well.
In summary:

1. Create a feature branch off of the `develop` branch with the name `feature/<my-feature-name>`.
2. Commit your changes and push to this feature branch.
3. When you are satisfied with your changes, initiate a [GitHub pull request](https://github.com/AP6YC/AdaptiveResonance.jl/pulls) (PR) to merge the feature branch with `develop`.
4. If the unit tests pass, the feature branch will first be merged with develop and then be deleted.
5. Releases will be periodically initiated from the `develop` branch and versioned onto the `master` branch.
6. Immediate bug fixes circumvent this process through a `hotfix` branch off of `master`.

## Development Details

### Documentation

These docs are currently hosted as a static site on the GitHub pages platform.
They are setup to be built and served in a separate branch `gh-pages` from the master/development branch of the project.

### Package Structure

The `AdaptiveResonance.jl` package has the following file structure:

```console
AdaptiveResonance
├── .github/workflows       // GitHub: workflows for testing and documentation.
├── data                    // Data: CI data location.
├── docs                    // Docs: documentation for the module.
│   └───src                 //      Documentation source files.
├── examples                // Source: example usage scripts.
├── src                     // Source: majority of source code.
│   ├───ART                 //      ART-based unsupervised modules.
│   └───ARTMAP              //      ARTMAP-based supervised modules.
├── test                    // Test: Unit, integration, and environment tests.
├── .appveyor               // Appveyor: Windows-specific coverage.
├── .gitattributes          // Git: LFS settings, languages, etc.
├── .gitignore              // Git: .gitignore for the whole project.
├── CODE_OF_CONDUCT.md      // Doc: the code of conduct for contributors.
├── CONTRIBUTING.md         // Doc: contributing guide (points to this page).
├── LICENSE                 // Doc: the license to the project.
├── Project.toml            // Julia: the Pkg.jl dependencies of the project.
└── README.md               // Doc: this document.
```

ART and ARTMAP algorithms are put into their own files within the `src/ART/` and `src/ARTMAP/` directories, respectively.
Both of these directories have an "index" file where each module is "included" (i.e., `src/ART/ART.jl`), which is in turn "included" in the package module file `src/AdaptiveResonance.jl`.

Abstract types and common structures/methods are included at the top of the package module file.
All public methods and structs (i.e., for the end user) are "exported" at the end of this file.

### ART Module Workflow

To write an ART module for this project, it will require the following:

1. A `train!` and `classify` method (within the module).
2. An keyword-options struct using the `Parameters.jl` macro `@with_kw` with assertions to keep the parameters within correct ranges.
3. Three constructors:
   1. A default constructor (i.e. `DDVFA()`).
   2. A keyword argument constructor (passing the kwargs to the options struct defined above).
   3. A constructor with the options struct passed itself.
4. Use of [common type aliases](@ref Type-Aliases) in method definitions.
5. An internal [`DataConfig`](@ref incremental_vs_batch) for setting up the data configuration, especially with `data_setup!` (`src/common.jl`).
6. An `update_iter` evaluation for each iteration (`src/common.jl`).
7. Inclusion to the correct ART index file (i.e., `src/ART/ART.jl`).
8. Exports of the names for the options and module constructors in the module definition (`src/AdaptiveResonance.jl`).

#### DataConfig

The original implementation of ART1 uses binary vectors, which have guaranteed separation between distinct vectors.
Real-valued ART modules, however, face the problem of permitting vectors to be arbitrarily close to one another.
Therefore, nearly every real-valued ART module uses [0, 1] normalization and complement-coding.
This is reflected in the `DataConfig` struct in the common file `src/common.jl`.

#### Type Aliases

In the pursuit of an architecture-agnostic implementation (i.e., support for both 32- and 64-bit systems), type aliases and other special Julia types are used in this project.

This module borrows a convention from the `StatsBase.jl` package by defining a variety of aliases for numerical types used throughout the package to standardize usage.
This has the benefits of readability and speed by explicitly
These are defined in `src/common.jl` and are currently as follows:

```julia
# Real-numbered aliases
const RealArray{T<:Real, N} = AbstractArray{T, N}
const RealVector{T<:Real} = AbstractArray{T, 1}
const RealMatrix{T<:Real} = AbstractArray{T, 2}

# Integered aliases
const IntegerArray{T<:Integer, N} = AbstractArray{T, N}
const IntegerVector{T<:Integer} = AbstractArray{T, 1}
const IntegerMatrix{T<:Integer} = AbstractArray{T, 2}

# Specifically floating-point aliases
const RealFP = Union{Float32, Float64}
```

In this package, data samples are always `Real`-valued (with the notable exception of [ART1](@ref incremental_vs_batch)), while class labels are integered.
Furthermore, independent class labels are always `Int` because of the [Julia native support](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Integers) for a given system's signed native integer type.

This project does not currently test for the support of [arbitrary precision arithmetic](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic) because learning algorithms *in general* do not have a significant need for precision beyond even 32-bit floats.

## Authors

If you simply have suggestions for improvement, Sasha Petrenko (<sap625@mst.edu>) is the current developer and maintainer of the AdaptiveResonance.jl package, so please feel free to reach out with thoughts and questions.
