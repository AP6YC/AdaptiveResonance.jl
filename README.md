# AdaptiveResonance

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

- [AdaptiveResonance](#adaptiveresonance)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Contributing](#contributing)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Implemented Modules](#implemented-modules)
  - [Structure](#structure)
  - [History](#history)
  - [Acknowledgements](#acknowledgements)
    - [Authors](#authors)
    - [Software](#software)
    - [Datasets](#datasets)
  - [License](#license)

## Overview

Adaptive Resonance Theory (ART) is a neurocognitive theory of how recurrent cellular networks can learn distributed patterns without supervision.
As a theory, it provides coherent and consistent explanations of how real neural networks learn patterns through competition, and it predicts the phenomena of attention and expectation as central to learning.
In engineering, the theory has been applied to a myriad of algorithmic models for unsupervised machine learning, though it has been extended to supervised and reinforcement learning frameworks.
This package provides implementations of many of these algorithms in Julia for both scientific research and engineering applications.
Basic installation is outlined in [Installation](#installation), while a quickstart is provided in [Quickstart](#quickstart).
Detailed usage and examples are provided in the [documentation](https://ap6yc.github.io/AdaptiveResonance.jl/dev/).

## Contributing

If you have a question or concern, please raise an [issue][issues-url].
For more details on how to work with the project, propose changes, or even contribute code, please see the [Developer Notes][contrib-url] in the project's documentation.

In summary:

1. Questions and requested changes should all be made in the [issues][issues-url] page.
These are preferred because they are publicly viewable and could assist or educate others with similar issues or questions.
2. For changes, this project accepts pull requests (PRs) from `feature/<my-feature>` branches onto the `develop` branch using the [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) methodology.
If unit tests pass and the changes are beneficial, these PRs are merged into `develop` and eventually folded into versioned releases.
3. The project follows the [Semantic Versioning](https://semver.org/) convention of `major.minor.patch` incremental versioning numbers.
Patch versions are for bug fixes, minor versions are for backward-compatible changes, and major versions are for new and incompatible usage changes.

## Installation

This project is distributed as a Julia package, available on [JuliaHub](https://juliahub.com/).
Its usage follows the usual Julia package installation procedure, interactively:

```julia
] add AdaptiveResonance
```

or programmatically:

```julia
using Pkg
Pkg.add("AdaptiveResonance")
```

You may also add the package directly from GitHub to get the latest changes between releases:

```julia
] add https://github.com/AP6YC/AdaptiveResonance.jl
```

## Quickstart

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

For more advanced users, options for the modules are contained in `Parameters.jl` structs.
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

## Implemented Modules

This project has implementations of the following ART (unsupervised) and ARTMAP (supervised) modules:

- ART
  - **FuzzyART**: Fuzzy ART
  - **DVFA**: Dual Vigilance Fuzzy ART
  - **DDVFA**: Distributed Dual Vigilance Fuzzy ART
- ARTMAP
  - **SFAM**: Simplified Fuzzy ARTMAP
  - **FAM**: Fuzzy ARTMAP
  - **DAM**: Default ARTMAP

Because each of these modules is a framework for many variants in the literature, this project also implements these [variants](https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/modules/) by changing their module [options](https://ap6yc.github.io/AdaptiveResonance.jl/dev/man/guide/#art_options).

In addition to these modules, this package contains the following accessory methods:

- **ARTSCENE**: the ARTSCENE algorithm's multiple-stage filtering process is implemented as `artscene_filter`. Each filter stage is exported if further granularity is required.
- **performance**: classification accuracy is implemented as `performance`
- **complement_code**: complement coding is implemented with `complement_code`.
However, training and classification methods complement code their inputs unless they are passed `preprocessed=true`.
- **linear_normalization**: the first step to complement coding, `linear_normalization` normalizes input arrays within [0, 1].

## Structure

The following file tree summarizes the project structure:

```console
AdaptiveResonance
├── .github/workflows       // GitHub: workflows for testing and documentation.
├── docs                    // Docs: documentation for the module.
│   ├─── examples           //      DemoCards documentation examples.
│   └─── src                //      Documentation source files.
├── paper                   // JOSS: journal paper and citations.
├── src                     // Source: majority of source code.
│   ├─── ART                //      ART-based unsupervised modules.
│   └─── ARTMAP             //      ARTMAP-based supervised modules.
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

## History

- 7/10/2020 - Begin project.
- 11/3/2020 - Complete baseline modules and tests.
- 2/8/2021 - Formalize usage documentation.
- 10/13/2021 - Initiate GitFlow contribution.

## Acknowledgements

### Authors

This package is developed and maintained by [Sasha Petrenko](https://github.com/AP6YC) with sponsorship by the [Applied Computational Intelligence Laboratory (ACIL)](https://acil.mst.edu/). This project is supported by grants from the [Night Vision Electronic Sensors Directorate](https://c5isr.ccdc.army.mil/inside_c5isr_center/nvesd/), the [DARPA Lifelong Learning Machines (L2M) program](https://www.darpa.mil/program/lifelong-learning-machines), [Teledyne Technologies](http://www.teledyne.com/), and the [National Science Foundation](https://www.nsf.gov/).
The material, findings, and conclusions here do not necessarily reflect the views of these entities.

The users [@aaronpeikert](https://github.com/aaronpeikert), [@hayesall](https://github.com/hayesall), and [@markNZed](https://github.com/markNZed) have graciously contributed their time with reviews and feedback that has greatly improved the project.

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

1. UCI machine learning repository:
<http://archive.ics.uci.edu/ml>

2. Fundamental Clustering Problems Suite (FCPS):
<https://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1>

3. Datasets package:
<https://www.researchgate.net/publication/239525861_Datasets_package>

4. Clustering basic benchmark:
<http://cs.uef.fi/sipu/datasets>

## License

This software is openly maintained by the ACIL of the Missouri University of Science and Technology under the [MIT License](LICENSE).
