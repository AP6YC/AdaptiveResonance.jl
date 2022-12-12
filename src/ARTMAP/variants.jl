"""
    variants.jl

Description:
    Includes convenience constructors for common variants of various ARTMAP modules.
"""

"""
Constructs a Default ARTMAP module using a SFAM module using Default ARTMAP's choice-by-difference activation function.

# References:
1. G. P. Amis and G. A. Carpenter, 'Default ARTMAP 2,' IEEE Int. Conf. Neural Networks - Conf. Proc., vol. 2, no. September 2007, pp. 777-782, Mar. 2007, doi: 10.1109/IJCNN.2007.4371056.
"""
function DAM(;kwargs...)
    return SFAM(;choice_by_difference=true, kwargs...)
end
