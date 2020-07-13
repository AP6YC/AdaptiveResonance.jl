module AdaptiveResonance

include("funcs.jl")

export doGreet, my_f, greet

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