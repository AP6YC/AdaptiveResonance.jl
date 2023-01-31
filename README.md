[![adaptiveresonance-header](docs/src/assets/header.png)][docs-dev-url]

A Julia package for Adaptive Resonance Theory (ART) algorithms.

| **Documentation**  | **Testing Status** | **Coverage** | **Reference** |
|:------------------:|:----------------:|:------------:|:-------------:|
| [![Stable][docs-stable-img]][docs-stable-url] | [![Build Status][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] | [![DOI][joss-img]][joss-url] |
| [![Dev][docs-dev-img]][docs-dev-url] | [![Build Status][appveyor-img]][appveyor-url] | [![Coveralls][coveralls-img]][coveralls-url] | [![DOI][zenodo-img]][zenodo-url] |
| **Documentation Build** | **JuliaHub Status** | **Dependents** | **Release** |
| [![Documentation][doc-status-img]][doc-status-url] | [![pkgeval][pkgeval-img]][pkgeval-url] | [![deps][deps-img]][deps-url] | [![version][version-img]][version-url] |

[joss-img]: https://joss.theoj.org/papers/10.21105/joss.03671/status.svg
[joss-url]: https://doi.org/10.21105/joss.03671

[doc-status-img]: https://github.com/AP6YC/AdaptiveResonance.jl/actions/workflows/Documentation.yml/badge.svg
[doc-status-url]: https://github.com/AP6YC/AdaptiveResonance.jl/actions/workflows/Documentation.yml

[zenodo-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.5748453.svg
[zenodo-url]: https://doi.org/10.5281/zenodo.5748453

[deps-img]: https://juliahub.com/docs/AdaptiveResonance/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/AdaptiveResonance/Sm0We?t=2

[version-img]: https://juliahub.com/docs/AdaptiveResonance/version.svg
[version-url]: https://juliahub.com/ui/Packages/AdaptiveResonance/Sm0We

[pkgeval-img]: https://juliahub.com/docs/AdaptiveResonance/pkgeval.svg
[pkgeval-url]: https://juliahub.com/ui/Packages/AdaptiveResonance/Sm0We

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://AP6YC.github.io/AdaptiveResonance.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://AP6YC.github.io/AdaptiveResonance.jl/dev

[ci-img]: https://github.com/AP6YC/AdaptiveResonance.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/AP6YC/AdaptiveResonance.jl/actions?query=workflow%3ACI

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/AP6YC/AdaptiveResonance.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/AP6YC/AdaptiveResonance-jl

[codecov-img]: https://codecov.io/gh/AP6YC/AdaptiveResonance.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/AP6YC/AdaptiveResonance.jl

[coveralls-img]: https://coveralls.io/repos/github/AP6YC/AdaptiveResonance.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/AP6YC/AdaptiveResonance.jl?branch=master

[issues-url]: https://github.com/AP6YC/AdaptiveResonance.jl/issues
[contrib-url]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/contributing/

Please read the [documentation](https://ap6yc.github.io/AdaptiveResonance.jl/dev/) for detailed usage and tutorials.

## Contents

- [Contents](#contents)
- [Overview](#overview)
- [Usage](#usage)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Implemented Modules](#implemented-modules)
  - [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
  - [Authors](#authors)
  - [Funding](#funding)
  - [History](#history)
  - [Software](#software)
  - [Datasets](#datasets)
  - [License](#license)
  - [Citation](#citation)

## Overview

Adaptive Resonance Theory (ART) is a neurocognitive theory of how recurrent cellular networks can learn distributed patterns without supervision.
As a theory, it provides coherent and consistent explanations of how real neural networks learn patterns through competition, and it predicts the phenomena of attention and expectation as central to learning.
In engineering, the theory has been applied to a myriad of algorithmic models for unsupervised machine learning, though it has been extended to supervised and reinforcement learning frameworks.
This package provides implementations of many of these algorithms in Julia for both scientific research and engineering applications.
Basic installation is outlined in [Installation](#installation), while a quickstart is provided in [Quickstart](#quickstart).
Detailed usage and examples are provided in the [documentation](https://ap6yc.github.io/AdaptiveResonance.jl/dev/).

## Usage

### Installation

This project is distributed as a [Julia](https://julialang.org/) package, available on [JuliaHub](https://juliahub.com/), so you must first [install Julia](https://julialang.org/downloads/) on your system.
Its usage follows the usual [Julia package installation procedure](https://docs.julialang.org/en/v1/stdlib/Pkg/), interactively:

```julia-repl
julia> ]
(@v1.8) pkg> add AdaptiveResonance
```

or programmatically:

```julia-repl
julia> using Pkg
julia> Pkg.add("AdaptiveResonance")
```

You may also add the package directly from GitHub to get the latest changes between releases:

```julia-repl
julia> ]
(@v1.8) pkg> add https://github.com/AP6YC/AdaptiveResonance.jl
```

### Quickstart

Load the module with

```julia
using AdaptiveResonance
```

The stateful information of ART modules are structs with default constructures such as

```julia
art = DDVFA()
```

You can pass module-specific options during construction with keyword arguments such as

```julia
art = DDVFA(rho_ub=0.75, rho_lb=0.4)
```

For more advanced users, options for the modules are contained in [`Parameters.jl`](https://github.com/mauro3/Parameters.jl) structs.
These options can be passed keyword arguments before instantiating the model:

```julia
opts = opts_DDVFA(rho_ub=0.75, rho_lb=0.4)
art = DDVFA(opts)
```

Train and test the models with `train!` and `classify`:

```julia
# Unsupervised ART module
art = DDVFA()

# Supervised ARTMAP module
artmap = SFAM()

# Load some data
train_x, train_y, test_x, test_y = load_your_data()

# Unsupervised training and testing
train!(art, train_x)
y_hat_art = classify(art, test_x)

# Supervised training and testing
train!(artmap, train_x, train_y)
y_hat_artmap = classify(art, test_x)
```

`train!` and `classify` can accept incremental or batch data, where rows are features and columns are samples.

Unsupervised ART modules can also accommodate simple supervised learning where internal categories are mapped to supervised labels with the keyword argument `y`:

```julia
# Unsupervised ART module
art = DDVFA()
train!(art, train_x, y=train_y)
```

These modules also support retrieving the "best-matching unit" in the case of complete mismatch (i.e., the next-best category if the presented sample is completely unrecognized) with the keyword argument `get_bmu`:

```julia
# Get the best-matching unit in the case of complete mismatch
y_hat_bmu = classify(art, test_x, get_bmu=true)
```

### Implemented Modules

This project has implementations of the following ART (unsupervised) and ARTMAP (supervised) modules:

- ART
  - **[`FuzzyART`][1]**: Fuzzy ART
  - **[`DVFA`][2]**: Dual Vigilance Fuzzy ART
  - **[`DDVFA`][3]**: Distributed Dual Vigilance Fuzzy ART
- ARTMAP
  - **[`SFAM`][4]**: Simplified Fuzzy ARTMAP
  - **[`FAM`][5]**: Fuzzy ARTMAP

Because each of these modules is a framework for many variants in the literature, this project also implements these [variants](https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/modules/) by changing their module [options](https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/guide/#art_options).
Variants built upon these modules are:

- ART
  - **[`GammaNormalizedFuzzyART`][7]**: Gamma-Normalized FuzzyART (variant of FuzzyART).
- ARTMAP
  - **[`DAM`][6]**: Default ARTMAP (variant of SFAM).

[1]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.FuzzyART
[2]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.DVFA
[3]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.DDVFA
[4]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.SFAM
[5]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.FAM
[6]: https://ap6yc.github.io/AdaptiveResonance.jl/stable/man/full-index/#AdaptiveResonance.DAM-Tuple{opts_SFAM}
[7]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.GammaNormalizedFuzzyART-Tuple{opts_FuzzyART}

In addition to these modules, this package contains the following accessory methods:

- [**ARTSCENE**][21]: the ARTSCENE algorithm's multiple-stage filtering process is implemented as [`artscene_filter`][21]. Each filter stage is implemented internally if further granularity is required.
- [**performance**][22]: classification accuracy is implemented as [`performance`][22].
- [**complement_code**][23]: complement coding is implemented with [`complement_code`][23].
However, training and classification methods complement code their inputs unless they are passed `preprocessed=true`, indicating to the model that this step has already been done.
- [**linear_normalization**][24]: the first step to complement coding, [`linear_normalization`][24] normalizes input arrays within `[0, 1]`.

[21]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.artscene_filter-Union{Tuple{Array{T,%203}},%20Tuple{T}}%20where%20T%3C:AbstractFloat
[22]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.performance-Tuple{AbstractVector{T}%20where%20T%3C:Integer,%20AbstractVector{T}%20where%20T%3C:Integer}
[23]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.complement_code-Tuple{AbstractArray{T}%20where%20T%3C:Real}
[24]: https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/full-index/#AdaptiveResonance.linear_normalization-Tuple{AbstractMatrix{T}%20where%20T%3C:Real}

### Contributing

If you have a question or concern, please raise an [issue][issues-url].
For more details on how to work with the project, propose changes, or even contribute code, please see the [Developer Notes][contrib-url] in the project's documentation.

In summary:

1. Questions and requested changes should all be made in the [issues][issues-url] page.
These are preferred because they are publicly viewable and could assist or educate others with similar issues or questions.
2. For changes, this project accepts pull requests (PRs) from `feature/<my-feature>` branches onto the `develop` branch using the [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) methodology.
If unit tests pass and the changes are beneficial, these PRs are merged into `develop` and eventually folded into versioned releases throug a `release` branch that is merged with the `master` branch.
3. The project follows the [Semantic Versioning](https://semver.org/) convention of `major.minor.patch` incremental versioning numbers.
Patch versions are for bug fixes, minor versions are for backward-compatible changes, and major versions are for new and incompatible usage changes.

## Acknowledgements

### Authors

This package is developed and maintained by [Sasha Petrenko](https://github.com/AP6YC) with sponsorship by the [Applied Computational Intelligence Laboratory (ACIL)](https://acil.mst.edu/).
The users [@aaronpeikert](https://github.com/aaronpeikert), [@hayesall](https://github.com/hayesall), and [@markNZed](https://github.com/markNZed) have graciously contributed their time with reviews and feedback that has greatly improved the project.

### Funding

This project is supported by grants from the [Night Vision Electronic Sensors Directorate](https://c5isr.ccdc.army.mil/inside_c5isr_center/nvesd/), the [DARPA Lifelong Learning Machines (L2M) program](https://www.darpa.mil/program/lifelong-learning-machines), [Teledyne Technologies](http://www.teledyne.com/), and the [National Science Foundation](https://www.nsf.gov/).
The material, findings, and conclusions here do not necessarily reflect the views of these entities.

Research was sponsored by the Army Research Laboratory and was accomplished under
Cooperative Agreement Number W911NF-22-2-0209.
The views and conclusions contained in this document are
those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of
the Army Research Laboratory or the U.S. Government.
The U.S. Government is authorized to reproduce and
distribute reprints for Government purposes notwithstanding any copyright notation herein.

### History

- 7/10/2020 - Begin project.
- 11/3/2020 - Complete baseline modules and tests.
- 2/8/2021 - Formalize usage documentation.
- 10/13/2021 - Initiate GitFlow contribution.
- 5/4/2022 - [Acceptance to JOSS](https://doi.org/10.21105/joss.03671).
- 10/11/2022 - v0.6.0
- 12/15/2022 - v0.7.0
- 1/30/2023 - v0.8.0

### Software

Adaptive Resonance Theory has been developed in theory and in application by many research groups since the theory's conception, and so this project was not developed in a vacuum.
This project itself is built upon the wisdom and precedent of decades of previous work in ART in a variety of programming languages.
The code in this repository is inspired the following repositories:

- [ACIL Organization GitHub](https://github.com/ACIL-Group)
  - MATLAB
    - [DDVFA](https://github.com/ACIL-Group/DDVFA): Companion MATLAB implementation of distrubuted dual vigilance fuzzy ART.
    - [DVFA](https://github.com/ACIL-Group/DVFA): Companion MATLAB code for Dual Vigilance Fuzzy ART
    - [iCVI-toolbox](https://github.com/ACIL-Group/iCVI-toolbox): A MATLAB toolbox for incremental/batch cluster validity indices
    - [CVIFA](https://github.com/ACIL-Group/CVIFA): Companion MATLAB implementation of validity index-based vigilance test fuzzy ART.
    - [VAT-FA](https://github.com/ACIL-Group/VAT-FA): Companion MATLAB code for VAT + Fuzzy ART.
    - [BARTMAP-CF](https://github.com/ACIL-Group/BARTMAP-CF): Companion MATLAB code for BARTMAP-based collaborative filtering
  - Python
    - [NuART-Py](https://github.com/ACIL-Group/NuART-Py): An internal ACIL python package for ART neural networks.
    - [DVHA](https://github.com/ACIL-Group/DVHA): An python implementation of dual vigilance hypersphere ART.
- [Boston University's Cognitive and Neural Systems (CNS) Tech Lab](http://techlab.bu.edu/resources/software/C51.html)
- [Nanyang Technological University's Tan Ah Whee](ntu.edu.sg/home/asahtan/downloads.htm)
- [Bernabé Linares-Barranco](http://www2.imse-cnm.csic.es/~bernabe)
- [Marko Tscherepanow's LibTopoART](libtopoart.eu)
- [National University of Singapore's Lei Meng](https://github.com/Lei-Meng)
- [Daniel Tauritz's ART Clearinghouse](https://web.mst.edu/~tauritzd/art/)

### Datasets

Boilerplate clustering datasets are periodically used to test, verify, and provide example of the functionality of the package.

1. [UCI machine learning repository](http://archive.ics.uci.edu/ml)
2. [Fundamental Clustering Problems Suite (FCPS)](https://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1)
3. [Nejc Ilc's unsupervised datasets package](https://www.researchgate.net/publication/239525861_Datasets_package)
4. [Clustering basic benchmark](http://cs.uef.fi/sipu/datasets)

### License

This software is openly maintained by the ACIL of the Missouri University of Science and Technology under the [MIT License](LICENSE).

### Citation

This project has a [citation file](CITATION.cff) file that generates citation information for the package and corresponding JOSS paper, which can be accessed at the "Cite this repository button" under the "About" section of the GitHub page.

You may also cite this repository with the following BibTeX entry:

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
