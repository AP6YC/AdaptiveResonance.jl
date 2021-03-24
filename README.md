# AdaptiveResonance

A Julia package for Adaptive Resonance Theory (ART) algorithms.

| **Documentation**  | **Build Status** | **Coverage** |
|:------------------:|:----------------:|:------------:|
| [![Stable][docs-stable-img]][docs-stable-url] [![Dev][docs-dev-img]][docs-dev-url] | [![Build Status][ci-img]][ci-url] [![Build Status][appveyor-img]][appveyor-url] | [![Codecov][codecov-img]][codecov-url] [![Coveralls][coveralls-img]][coveralls-url] |

| **Dependents** | **Date** | **Status** |
|:--------------:|:--------:|:----------:|
| [![deps](https://juliahub.com/docs/AdaptiveResonance/deps.svg)](https://juliahub.com/ui/Packages/AdaptiveResonance/Sm0We?t=2) | [![version](https://juliahub.com/docs/AdaptiveResonance/version.svg)](https://juliahub.com/ui/Packages/AdaptiveResonance/Sm0We) | [![pkgeval](https://juliahub.com/docs/AdaptiveResonance/pkgeval.svg)](https://juliahub.com/ui/Packages/AdaptiveResonance/Sm0We) |
<!-- | [![Stable][docs-stable-img]][docs-stable-url] [![Dev][docs-dev-img]][docs-dev-url] | [![Build Status][travis-img]][travis-url] [![Build Status][appveyor-img]][appveyor-url] | [![Codecov][codecov-img]][codecov-url] [![Coveralls][coveralls-img]][coveralls-url] | -->

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://AP6YC.github.io/AdaptiveResonance.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://AP6YC.github.io/AdaptiveResonance.jl/dev

[ci-img]: https://github.com/AP6YC/AdaptiveResonance.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/AP6YC/AdaptiveResonance.jl/actions?query=workflow%3ACI
<!-- [travis-img]: https://travis-ci.com/AP6YC/AdaptiveResonance.jl.svg?branch=master -->
<!-- [travis-url]: https://travis-ci.com/AP6YC/AdaptiveResonance.jl -->

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/AP6YC/AdaptiveResonance.jl?svg=true
[appveyor-url]: https://ci.appveyor.com/project/AP6YC/AdaptiveResonance-jl

[codecov-img]: https://codecov.io/gh/AP6YC/AdaptiveResonance.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/AP6YC/AdaptiveResonance.jl

[coveralls-img]: https://coveralls.io/repos/github/AP6YC/AdaptiveResonance.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/AP6YC/AdaptiveResonance.jl?branch=master

[issues-url]: https://github.com/AP6YC/AdaptiveResonance.jl/issues
[contrib-url]: https://juliadocs.github.io/Documenter.jl/dev/contributing/
[discourse-tag-url]: https://discourse.julialang.org/tags/documenter
[gitter-url]: https://gitter.im/juliadocs/users

This package is developed and maintained by [Sasha Petrenko](https://github.com/AP6YC) with sponsorship by the [Applied Computational Intelligence Laboratory (ACIL)](https://acil.mst.edu/). This project is supported by grants from the [Night Vision Electronic Sensors Directorate](https://c5isr.ccdc.army.mil/inside_c5isr_center/nvesd/), the [DARPA Lifelong Learning Machines (L2M) program](https://www.darpa.mil/program/lifelong-learning-machines), [Teledyne Technologies](http://www.teledyne.com/), and the [National Science Foundation](https://www.nsf.gov/).
The material, findings, and conclusions here do not necessarily reflect the views of these entities.

Please read the [documentation](https://ap6yc.github.io/AdaptiveResonance.jl/dev/) for detailed usage and tutorials.

## Contents

- [AdaptiveResonance](#adaptiveresonance)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Structure](#structure)
  - [Contributing](#contributing)
  - [History](#history)
  - [Credits](#credits)
    - [Authors](#authors)
    - [Software](#software)
    - [Datasets](#datasets)
  - [License](#license)

## Overview

Adaptive Resonance Theory (ART) is a neurocognitive theory of how recurrent cellular networks can learn distributed patterns without supervision.
As a theory, it provides coherent and consistent explanations of how real neural networks learn patterns through competition, and it predicts the phenomena of attention and expectation as central to learning.
In engineering, the theory has been applied to a myriad of algorithmic models for unsupervised machine learning, though it has been extended to supervised and reinforcement learning frameworks.
This package provides implementations of many of these algorithms in Julia for both scientific research and engineering applications.
A quickstart is provided in [Installation](#Installation), while detailed usage and examples are provided in the [documentation](https://ap6yc.github.io/AdaptiveResonance.jl/dev/).

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

You may also add the package directly from GitHub to get the latest changes:

```julia
] add https://github.com/AP6YC/AdaptiveResonance.jl
```

## Structure

The following file tree summarizes the project structure:

```console
AdaptiveResonance
├── .github/workflows       // GitHub: workflows for testing and documentation.
├── data                    // Data: CI data location.
├── docs                    // Docs: documentation for the module.
│   └───src                 //      Documentation source files.
├── examples                // Source: example usage scripts.
├── src                     // Source: majority of source code.
│   ├───ART                 //      ART-based unsupervised modules.
│   ├───ARTMAP              //      ARTMAP-based supervised modules.
│   └───CVI                 //      Cluster validity indices.
├── test                    // Test: Unit, integration, and environment tests.
├── .appveyor               // Appveyor: Windows-specific coverage.
├── .gitignore              // Git: .gitignore for the whole project.
├── LICENSE                 // Doc: the license to the project.
├── Project.toml            // Julia: the Pkg.jl dependencies of the project.
└── README.md               // Doc: this document.
```

## Contributing

Please raise an [issue][issues-url].

## History

- 7/10/2020 - Begin project.
- 11/3/2020 - Complete baseline modules and tests.
- 2/8/2021 - Formalize usage documentation.

## Credits

### Authors

- Sasha Petrenko <sap625@mst.edu>

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

### Datasets

Boilerplate clustering datasets are periodically used to test, verify, and provide example of the functionality of the package.

1. UCI machine learning repository:
http://archive.ics.uci.edu/ml

2. Fundamental Clustering Problems Suite (FCPS):
https://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1

3. Datasets package:
https://www.researchgate.net/publication/239525861_Datasets_package

4. Clustering basic benchmark:
http://cs.uef.fi/sipu/datasets

## License

This software is openly maintained by the ACIL of the Missouri University of Science and Technology under the [MIT License](LICENSE).
