export my_f, foo, doGreet, greet


function foo(a, b)
    return a + b
end

my_f(x, y) = 2x + y

"""
    greet()

Print a hello world!

# Examples
```julia-repl
julia> greet()
Hello World!
```
"""
greet() = println("Hello World!")


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
    println("Hello ", name, "!")
end