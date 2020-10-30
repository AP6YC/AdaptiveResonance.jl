# AdaptiveResonance

A Julialang package for adaptive resonance theory (ART) algorithms.

| **Documentation**  | **Build Status** | **Coverage** |
|:------------------:|:----------------:|:------------:|
| [![Stable][docs-stable-img]][docs-stable-url] [![Dev][docs-dev-img]][docs-dev-url] | [![Build Status][travis-img]][travis-url] [![Build Status][appveyor-img]][appveyor-url] | [![Codecov][codecov-img]][codecov-url] [![Coveralls][coveralls-img]][coveralls-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://AP6YC.github.io/AdaptiveResonance.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://AP6YC.github.io/AdaptiveResonance.jl/dev

[travis-img]: https://travis-ci.com/AP6YC/AdaptiveResonance.jl.svg?branch=master
[travis-url]: https://travis-ci.com/AP6YC/AdaptiveResonance.jl

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


This package is developed and maintained by [Sasha Petrenko](https://github.com/AP6YC) with sponsorship by the [Applied Computationa Intelligence Laboratory (ACIL)](https://acil.mst.edu/). This project is supported from grants with the [Night Vision Electronic Sensors Directorate](https://c5isr.ccdc.army.mil/inside_c5isr_center/nvesd/), the [DARPA Lifelong Learning Machines (L2M) program](https://www.darpa.mil/program/lifelong-learning-machines), and [Teledyne Technologies](http://www.teledyne.com/).

# Contents

- [AdaptiveResonance](#adaptiveresonance)
- [Contents](#contents)
- [Structure](#structure)
- [Contributing](#contributing)
- [History](#history)
- [Credits](#credits)
  - [Authors:](#authors)
  - [Software](#software)
  - [Datasets](#datasets)
- [License](#license)

# Structure

The following file-tree summarizes the project structure:

```
AdaptiveResonance
├── docs                    // Docs: documentation for the module
│   └───src                 //      Documentation source files
├── src                     // Source: majority of source code
├── test                    // Test: Unit, integration, and environment tests
├── .appveyor               // Appveyor: Windows-specific coverage
├── .gitignore              // Git: .gitignore for the whole project
├── .travis.yml             // Travis: code coverage automation
├── LICENSE                 // Doc: the license to the project
├── Project.toml            // Julia: the Pkg.jl dependencies of the project
└── README.md               // Doc: this document
```

# Contributing

Please raise an [issue][issues-url].


# History

* 7/10/2020 - Begin project

# Credits

## Authors:
* Sasha Petrenko <sap625@mst.edu>

## Software

The code in this repository is inspired by and occasionally leverages the following repositories:

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


## Datasets

1. UCI machine learning repository:
http://archive.ics.uci.edu/ml

2. Fundamental Clustering Problems Suite (FCPS):
https://www.uni-marburg.de/fb12/arbeitsgruppen/datenbionik/data?language_sync=1

3. Datasets package:
https://www.researchgate.net/publication/239525861_Datasets_package

4. Clustering basic benchmark:
http://cs.uef.fi/sipu/datasets

# License

This software is openly maintained by the ACIL of the Missouri University of Science and Technology under the [MIT License](LICENSE).

<!--Old CI token code:

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://AP6YC.github.io/AdaptiveResonance.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://AP6YC.github.io/AdaptiveResonance.jl/dev)
[![Build Status](https://travis-ci.com/AP6YC/AdaptiveResonance.jl.svg?branch=master)](https://travis-ci.com/AP6YC/AdaptiveResonance.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/AP6YC/AdaptiveResonance.jl?svg=true)](https://ci.appveyor.com/project/AP6YC/AdaptiveResonance-jl)
[![Codecov](https://codecov.io/gh/AP6YC/AdaptiveResonance.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AP6YC/AdaptiveResonance.jl)
[![Coveralls](https://coveralls.io/repos/github/AP6YC/AdaptiveResonance.jl/badge.svg?branch=master)](https://coveralls.io/github/AP6YC/AdaptiveResonance.jl?branch=master) -->