"""
foo(a, b)

The ubiquitous foo function to verify basic functionality in tests.
    Returns a + b

# Examples
```julia-repl
julia> foo(1, 2)
3
```
"""
function foo(a, b)
    return a + b
end


"""
    my_f(x, y)

Basic inline function to verify basic functionality in tests.
    Returns 2 * x + y

# Examples
```julia-repl
julia> my_f(2, 1)
5
```
"""
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