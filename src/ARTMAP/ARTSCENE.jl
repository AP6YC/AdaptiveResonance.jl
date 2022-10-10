"""
    ARTSCENE.jl

Description:
    All of the visual filter functions for the ARTSCENE algorithm.
"""

# --------------------------------------------------------------------------- #
# DEPENDENCIES
# --------------------------------------------------------------------------- #

using Distributed
using SharedArrays

# --------------------------------------------------------------------------- #
# FUNCTIONS
# --------------------------------------------------------------------------- #

"""
ARTSCENE Stage 1: Color-to-gray image transformation.
"""
function color_to_gray(image::RealArray)
    # Treat the image as a column-major array, cast to grayscale
    dim, n_row, n_column = size(image)
    return [sum(image[:,i,j])/3 for i=1:n_row, j=1:n_column]
end

"""
Surround kernel S function for ARTSCENE Stage 2.
"""
function surround_kernel(i::Integer, j::Integer, p::Integer, q::Integer, scale::Integer)
    return 1/(2*pi*scale^2)*MathConstants.e^(-((i-p)^2 + (j-q)^2)/(2*scale^2))
end

"""
Time rate of change of LGN network (ARTSCENE Stage 2).
"""
function ddt_x(x::RealArray, image::RealArray, sigma_s::RealArray, distributed::Bool)
    n_row, n_column = size(x)
    n_g = length(sigma_s)
    kernel_r = 5

    # dx = zeros(n_row, n_column, 4)
    # for g = 1:n_g
    if distributed
        dx = SharedArray{Float64, 3}((n_row, n_column, n_g))
        @sync @distributed for g = 1:n_g
        for i = 1:n_row
        for j = 1:n_column
            # Compute the surround kernel
            kernel_h = max(1, i-kernel_r):min(n_row, i + kernel_r)
            kernel_w = max(1, j-kernel_r):min(n_row, j + kernel_r)
            S_ijg_I = sum([surround_kernel(i, j, p, q, sigma_s[g])*image[p, q]
                        for p in kernel_h, q in kernel_w])
            # Compute the enhanced contrast
            dx[i,j,g] = - x[i,j,g] + (1 - x[i,j,g])*image[i,j] - (1 + x[i,j,g])*S_ijg_I
        end
        end
        end
    end
    return dx
end

"""
ARTSCENE Stage 2: Constrast normalization.
"""
function contrast_normalization(image::RealArray ; distributed::Bool=true)
    # All scale parameters
    sigma_s = [1, 4, 8, 12]
    n_g = length(sigma_s)

    # Number if iterations to settle on contrast result
    n_iterations = 4
    # Get the shape of the image
    n_row, n_column = size(image)
    x = zeros(n_row, n_column, n_g)

    for g = 1:n_g
        x[:,:, g] = deepcopy(image)
    end

    for i = 1:n_iterations
        x += ddt_x(x, image, sigma_s, distributed)
    end

    return x
end

"""
Oriented, elongated, spatially offset kernel G for ARTSCENE Stage 3.
"""
function oriented_kernel(i::Integer, j::Integer, p::Integer, q::Integer, k::Integer, sigma_h::Real, sigma_v::Real ; sign::AbstractString="plus")
    m = sin(pi*k/4)
    n = cos(pi*k/4)

    if sign == "plus"
        G = (1/(2*pi*sigma_h*sigma_v)*
            MathConstants.e^(-0.5*((((p-i+m)*n-(q-j+n)*m)/sigma_h)^2
                                  +(((p-i+m)*m+(q-j+n)*n)/sigma_v)^2)))
    elseif sign == "minus"
        G = (1/(2*pi*sigma_h*sigma_v)*
            MathConstants.e^(-0.5*((((p-i-m)*n-(q-j-n)*m)/sigma_h)^2
                                  +(((p-i-m)*m+(q-j-n)*n)/sigma_v)^2)))
    else
        throw("Incorrect sign option for oriented kernel function")
    end

    return G
end

"""
Shunting equation for ARTSCENE Stage 3.
"""
function ddt_y(y::RealArray, X_plus::RealArray, X_minus::RealArray, alpha::Real, distributed::Bool)
    # n_row, n_column = size(x) # TODO: SOURCE OF WRONGNESS
    n_row, n_column = size(y)
    n_k = 4
    sigma_v = [0.25, 1, 2, 3]
    sigma_h = [0.75, 3, 6, 9]
    n_g = length(sigma_v)
    kernel_r = 5

    # dy = zeros(n_row, n_column, n_k, n_g)
    # for k = 1:n_k
    if distributed
        dy = SharedArray{Float64, 4}((n_row, n_column, n_k, n_g))
        @sync @distributed for k = 1:n_k
        for g = 1:n_g
        for i = 1:n_row
        for j = 1:n_column
            # Compute the surround kernel
            kernel_h = max(1, i-kernel_r):min(n_row, i + kernel_r)
            kernel_w = max(1, j-kernel_r):min(n_row, j + kernel_r)
            Gp = [oriented_kernel(i, j, p, q, k-1, sigma_h[g], sigma_v[g], sign="plus")
                for p in kernel_h, q in kernel_w]
            Gm = [oriented_kernel(i, j, p, q, k-1, sigma_h[g], sigma_v[g], sign="minus")
                for p in kernel_h, q in kernel_w]
            dy[i,j,g,k] = (-alpha*y[i,j,g,k]
                        + (1-y[i,j,g,k])*sum(X_plus[kernel_h, kernel_w, g].*Gp
                                            + X_minus[kernel_h, kernel_w, g].*Gm)
                        - (1+y[i,j,g,k])*sum(X_plus[kernel_h, kernel_w, g].*Gm
                                            + X_minus[kernel_h, kernel_w, g].*Gp))
        end
        end
        end
        end
    end
    return dy
end

"""
ARTSCENE Stage 3: Contrast-sensitive oriented filtering.
"""
function contrast_sensitive_oriented_filtering(image::RealArray, x::RealArray ; distributed::Bool=true)
    # Get the size of the field
    n_row, n_column = size(x)

    # Parameters
    n_g = 4             # Number of scales
    n_k = 4             # Number of orientations
    alpha = 1           # Passive decay rate
    n_iterations = 4    # Number if iterations to settle on contrast result

    # Compute the LGN ON-cell and OFF-cell output signals
    X_plus = [max(0, x[i,j,g]) for i=1:n_row, j=1:n_column, g=1:n_g]
    X_minus = [max(0, -x[i,j,g]) for i=1:n_row, j=1:n_column, g=1:n_g]

    # Get the shape of the image
    n_row, n_column = size(x)
    y = zeros(n_row, n_column, n_g, n_k)
    for k = 1:n_k
        y[:,:,:,k] = deepcopy(x)
    end

    for _ = 1:n_iterations
        y += ddt_y(y, X_plus, X_minus, alpha, distributed)
    end

    return y
end

"""
ARTSCENE Stage 4: Contrast-insensitive oriented filtering.
"""
function contrast_insensitive_oriented_filtering(y::RealArray)
    n_row, n_column, n_g, n_k = size(y)

    # Compute the LGN ON-cell and OFF-cell output signals
    Y_plus = [max(0, y[i,j,g,k]) for i=1:n_row, j=1:n_column, g=1:n_g, k=1:n_k]
    Y_minus = [max(0, -y[i,j,g,k]) for i=1:n_row, j=1:n_column, g=1:n_g, k=1:n_k]

    return Y_plus + Y_minus
end

"""
Competition kernel for ARTSCENE: Stage 5.
"""
function competition_kernel(l::Integer, k::Integer ; sign::AbstractString="plus")

    if sign == "plus"
        g = ( 1/(0.5*sqrt(2*pi))*MathConstants.e^(-0.5*((l-k)/0.5)^2) )
    elseif sign == "minus"
        g = ( 1/(sqrt(2*pi))*MathConstants.e^(-0.5*(l-k)^2) )
    else
        throw("Incorrect sign option for oriented kernel function")
    end

    return g
end

"""
Time rate of change for ARTSCENE: Stage 5.
"""
function ddt_z(z::RealArray ; distributed::Bool=true)
    n_row, n_column, n_g, n_k = size(z)
    kernel_r = 5

    # dz = zeros(n_row, n_column, n_k, n_g)
    # for k = 1:n_k
    if distributed
        dz = SharedArray{Float64, 4}((n_row, n_column, n_k, n_g))
        @sync @distributed for k = 1:n_k
        for g = 1:n_g
        for i = 1:n_row
        for j = 1:n_column
            zgp = sum([z[i,j,g,l]*competition_kernel(l,k,sign="plus") for l = 1:n_g])
            zgm = sum([z[i,j,g,l]*competition_kernel(l,k,sign="minus") for l = 1:n_g])
            dz[i,j,g,k] = (- z[i,j,g,k]
                        + (1 - z[i,j,g,k]*zgp)
                        - (1 + z[i,j,g,k]*zgm))
        end
        end
        end
        end
    end

    return dz
end

"""
ARTSCENE Stage 5: Orientation competition at the same position.
"""
function orientation_competition(z::RealArray)

    # Parameters
    n_iterations = 4    # Number if iterations to settle on contrast result

    # Get the shape of the image
    # n_row, n_column, n_g, n_k = size(z)
    # Z = zeros(n_row, n_column, n_g, n_k)
    # for k = 1:n_k
    #     Z[:,:,:,k] = deepcopy(z)
    # end

    for _ = 1:n_iterations
        z += ddt_z(z)
    end

    return z
end

"""
ARTSCENE Stage 6: Create patch feature vectors.
"""
function patch_orientation_color(z::RealArray, image::RealArray)
    n_i, n_j, n_g, n_k = size(z)
    patch_i = 4
    patch_j = 4
    n_colors = 3
    n_patches = patch_i * patch_j
    size_i = n_i / patch_i
    size_j = n_j / patch_j
    size_patch = size_i * size_j
    O = zeros(patch_i, patch_j, n_g, n_k)
    C = zeros(patch_i, patch_j, n_colors)
    for p_i = 1:patch_i
        for p_j = 1:patch_j
            # Get the correct range objects for the grid
            i_range = Integer(floor(size_i*(p_i-1)+1)):Integer(floor(size_i*p_i))
            j_range = Integer(floor(size_j*(p_j-1)+1)):Integer(floor(size_j*p_j))
            # Compute the color averages
            for c = 1:n_colors
                C[p_i,p_j,c] = 1/size_patch*sum(image[c, i_range, j_range])
            end
            # Compute the orientation averages
            for k = 1:4
                for g = 1:4
                    O[p_i, p_j, k, g] = 1/size_patch * sum(z[i_range, j_range, k, g])
                end
            end
        end
    end
    return O, C
end

"""
Process the full artscene filter toolchain on an image.
"""
function artscene_filter(raw_image::Array{T, 3} ;  distributed::Bool=true) where {T<:Real}

    # Get the number of workers
    n_processes = nprocs()
    n_workers = nworkers()
    @debug "Processes: $n_processes, Workers: $n_workers"

    # Random image
    image_size = size(raw_image)
    image_type =  typeof(raw_image)
    @debug "Original: Size = $image_size, Type = $image_type"

    # Stage 1: Grayscale
    image = color_to_gray(raw_image)
    image_size = size(image)
    image_type = typeof(image)
    @debug "Stage 1 Complete: Grayscale: Size = $image_size, Type = $image_type"

    # Stage 2: Contrast normalization
    x = contrast_normalization(image, distributed=true)
    image_size = size(x)
    image_type = typeof(x)
    @debug "Stage 2 Complete: Contrast: Size = $image_size, Type = $image_type"

    # Stage 3: Contrast-sensitive oriented filtering
    y = contrast_sensitive_oriented_filtering(image, x)
    image_size = size(y)
    image_type = typeof(y)
    @debug "Stage 3 Complete: Sensitive Oriented: Size = $image_size, Type = $image_type"

    # Stage 4: Contrast-insensitive oriented filtering
    z = contrast_insensitive_oriented_filtering(y)
    image_size = size(z)
    image_type = typeof(z)
    @debug "Stage 4 Complete: Insensitive Oriented: Size = $image_size, Type = $image_type"

    # Stage 5: Orientation competition
    z = orientation_competition(z)
    image_size = size(z)
    image_type = typeof(z)
    @debug "Stage 5 Complete: Orientation Competition: Size = $image_size, Type = $image_type"

    # *Stage 6*: Compute patch vectors (orientation and color)
    O, C = patch_orientation_color(z, raw_image)
    @debug "Stage 6 Complete"

    return O, C
end
