module AdaptiveResonance

using Parameters
using CUDAapi
if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    import CuArrays		# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end


include("funcs.jl")


@with_kw mutable struct Args
    rho_ub
    rho_lb
    alpha
    beta
    w_init
    max_epochs
    shuffle::Bool = false
    random_seed::Int64 = 1234.5678
end

export doGreet, my_f, greet

"""
    ART()


"""

"""
    greet()

Prints a hello world!

# Examples
```julia-repl
julia> greet()
Hello World!
```
"""
greet() = print("Hello World!")


"""
    doGreet(name)

Greets the name given.

# Examples
```julia-repl
julia> doGreet("Julia")
Hello Julia!
```
"""
function doGreet(name)
    print("Hello ", name, "!")
end

end # module