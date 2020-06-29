using BenchmarkTools
a, b = zeros(Float32, 10^7), rand(Float32, 10^7)
# c_b = @belapsed cmap($sin, $a, $b)
# cxx_b = @belapsed cxxmap($sin, $a, $b)
# python_b = @elapsed pymap(sin, a, b);
# julia_b = @belapsed map!($sin, $a, $b);
# timings = [python_b, c_b, cxx_b, julia_b]
julia_b = @belapsed map!($sin, $a, $b);
timings = [julia_b]
