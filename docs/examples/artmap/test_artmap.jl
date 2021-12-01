# ---
# title: Write your demo in julia
# id: artmap_example
# cover: ../assets/art.png
# date: 2020-09-13
# author: "[Sasha Petrenko](https://github.com/AP6YC)"
# julia: 1.6
# description: This demo shows you how to write your demo in julia
# ---

# Different from markdown format demo, source demo files are preprocessed so that it generates:
#
# 1. assets such as cover image
# 2. julia source file
# 3. mardkown file
# 4. jupyter notebook file
#
# Links to nbviewer and source files are provided as well.

# !!! note
#     This entails every source file being executed three times: 1) asset generation, 2) notebook
#     generation, and 3) Documenter HTML generation. If the generation time is too long for your
#     application, you could then choose to write your demo in markdown format. Or you can disable
#     the notebook generation via the `notebook` keyword in your demos, this helps reduce the
#     runtime to 2x.

#
# The conversions from source demo files to `.jl`, `.md` and `.ipynb` files are mainly handled by
# [`Literate.jl`](https://github.com/fredrikekre/Literate.jl). The syntax to control/filter the
# outputs can be found [here](https://fredrikekre.github.io/Literate.jl/stable/fileformat/)

x = 1//3
y = 2//5
x + y

#-