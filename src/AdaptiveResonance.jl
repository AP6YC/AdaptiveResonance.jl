module AdaptiveResonance

include("funcs.jl")

export doGreet, my_f, greet

greet() = print("Hello World!")

function doGreet(name)
    print("Hello ", name, "!")
end

end # module