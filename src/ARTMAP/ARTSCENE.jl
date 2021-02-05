using Distributed
using SharedArrays
# using Images
# using Logging

# Set the log level
# LogLevel(Logging.Info)

# using ImageIO
# addprocs(4)
# @everywhere using SharedArrays
# @everywhere begin

# Stage 1: Color-to-gray image transformation
function color_to_gray(image::Array)
    # Treat the image as a column-major array, cast to grayscale
    dim, n_row, n_column = size(image)
    return [sum(image[:,i,j])/3 for i=1:n_row, j=1:n_column]
    # return Gray.(image)
end

# Surround kernel S function for Stage 2
function surround_kernel(i::Int, j::Int, p::Int, q::Int, scale::Int)
# @everywhere function surround_kernel(i::Int, j::Int, p::Int, q::Int, scale::Int)
    return 1/(2*pi*scale^2)*MathConstants.e^(-((i-p)^2 + (j-q)^2)/(2*scale^2))
end

# Time rate of change of LGN network (Stage 2)
# function ddt_x(x::SharedArray, image::Array, sigma_s::Array)
function ddt_x(x::Array, image::Array, sigma_s::Array, distributed::Bool)
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

# Stage 2: Constrast normalization
function contrast_normalization(image::Array ; distributed::Bool=true)
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

# Oriented, elongated, spatially offset kernel G for Stage 3
function oriented_kernel(i::Int, j::Int, p::Int, q::Int, k::Int, sigma_h::Real, sigma_v::Real ; sign::String="plus")
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

# Shunting equation for Stage 3
function ddt_y(y::Array, X_plus::Array, X_minus::Array, alpha::Real, distributed::Bool)
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

# Stage 3: Contrast-sensitive oriented filtering
function contrast_sensitive_oriented_filtering(image::Array, x::Array ; distributed::Bool=true)
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

    for i = 1:n_iterations
        y += ddt_y(y, X_plus, X_minus, alpha, distributed)
    end

    return y

end

# Stage 4: Contrast-insensitive oriented filtering
function contrast_insensitive_oriented_filtering(y::Array)

    n_row, n_column, n_g, n_k = size(y)

    # Compute the LGN ON-cell and OFF-cell output signals
    Y_plus = [max(0, y[i,j,g,k]) for i=1:n_row, j=1:n_column, g=1:n_g, k=1:n_k]
    Y_minus = [max(0, -y[i,j,g,k]) for i=1:n_row, j=1:n_column, g=1:n_g, k=1:n_k]

    return Y_plus + Y_minus
end

function competition_kernel(l::Int, k::Int ; sign::String="plus")

    if sign == "plus"
        g = ( 1/(0.5*sqrt(2*pi))*MathConstants.e^(-0.5*((l-k)/0.5)^2) )
    elseif sign == "minus"
        g = ( 1/(sqrt(2*pi))*MathConstants.e^(-0.5*(l-k)^2) )
    else
        throw("Incorrect sign option for oriented kernel function")
    end

    return g
end

function ddt_z(z::Array)
    n_row, n_column, n_g, n_k = size(z)
    kernel_r = 5

    # dz = zeros(n_row, n_column, n_k, n_g)
    # for k = 1:n_k
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

    return dz
end

# Stage 5: Orientation competition at the same position
function orientation_competition(z::Array)

    # Parameters
    n_iterations = 4    # Number if iterations to settle on contrast result

    # Get the shape of the image
    # n_row, n_column, n_g, n_k = size(z)
    # Z = zeros(n_row, n_column, n_g, n_k)
    # for k = 1:n_k
    #     Z[:,:,:,k] = deepcopy(z)
    # end

    for i = 1:n_iterations
        # Z += ddt_z(z)
        z += ddt_z(z)
    end

    return z
end

function patch_orientation_color(z::Array, image::Array)
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
            i_range = Int(size_i*(p_i-1)+1):Int(size_i*p_i)
            j_range = Int(size_j*(p_j-1)+1):Int(size_j*p_j)
            # Compute the color averages
            for c = 1:n_colors
                C[p_i,p_j,c] = 1/size_patch*sum(image[c, i_range, j_range])
            end
            # Compute the orientation averages
            for k = 1:4
                for g = 1:4
                    O[p_i,p_j,k,g] = 1/size_patch * sum(z[i_range, j_range, k, g])
                end
            end
        end
    end
    return O, C
end

# end

# include("../src/ARTMAP/ARTSCENE.jl")
# Random image
# image = rand(608, 608, 3)
# image_path = "E:\\dev\\mount\\200630_ForDistribution\\Cylinder1\\scene\\0070.png"
# # raw_image = load("data/image.png")
# raw_image = load(image_path)
# raw_image = imresize(raw_image, ratio=1/4)
# matrix_raw_image = convert(Array{Float64}, channelview(raw_image))

# Stage 1: Grayscale
# raw_image = convert(Array{Float64, 3}(undef, n_row, n_column, n_color), raw_image)
# image = color_to_gray(raw_image)
# image = convert(Array{Float64}, image)
# image = color_to_gray(matrix_raw_image)

# @info "Stage 1 Done"

# # Stage 2: Contrast normalization
# x = contrast_normalization(image)
# @info "Stage 2 Done"

# # Stage 3: Contrast-sensitive oriented filtering
# y = contrast_sensitive_oriented_filtering(image, x)
# @info "Stage 3 Done"

# # Stage 4: Contrast-insensitive oriented filtering
# z = contrast_insensitive_oriented_filtering(y)
# @info "Stage 4 Done"

# # Stage 5: Orientation competition
# z = orientation_competition(z)
# @info "Stage 5 Done"

# # *Stage 6*: Compute patch vectors (orientation and color)
# O, C = patch_orientation_color(z, matrix_raw_image)

# println(size(z))

# rmprocs(workers())

# # Stage 1: Color-to-gray image transformation
# function color_to_gray(image::Array)
#     # Treat the image as a column-major array, cast to grayscale
#     # Gray{Float64}.(image)
#     n_row, n_column, dim = size(image)
#     return [sum(image[i,j,:]) for i=1:n_row, j=1:n_column]
# end

# # Surround kernel S function for Stage 2
# function surround_kernel(i::Int, j::Int, p::Int, q::Int, scale::Int)
#     return 1/(2*pi*scale^2)*MathConstants.e^(-((i-p)^2 + (j-q)^2)/(2*scale^2))
# end

# # function get_S(n_row, n_column)
# #     scales = [1, 4, 8, 12]
# #     S = zeros(n_row, n_column, length(scales))
# #     for i = 1:n_row
# #         for j = 1:n_column
# #             S = [surround_kernel(i, j, p, q, scales[g]) for p=1:n_row, q=1:n_column]
# #         end
# #     end
# # end

# # Time rate of change of LGN network (Stage 2)
# function ddt_x(x::Array, image::Array, scales::Array)
#     n_row, n_column = size(x)
#     n_g = length(scales)
#     dx = zeros(n_row, n_column, 4)

#     for i = 1:n_row
#         for j = 1:n_column
#             for g = 1:n_g
#                 # Compute the surround kernel
#                 S_ijg_I = [surround_kernel(i, j, p, q, scales[g])*image[p, q]
#                            for p=1:n_row, q=1:n_column]
#                 # Compute the enhanced contrast
#                 dx[i,j,g] = -x[i,j,g]
#                             + (1 - x[i,j,g])*image[i,j]
#                             - (1 + x[i,j,g])*sum(S_ijg_I)
#             end
#         end
#     end
#     return dx
# end

# # Stage 2: Constrast normalization
# function contrast_normalization(image)
#     # All scale parameters
#     scales = [1, 4, 8, 12]
#     n_g = length(scales)

#     # Number if iterations to settle on contrast result
#     n_iterations = 5

#     # Get the shape of the image
#     n_row, n_column = size(image)
#     x = zeros(n_row, n_column, n_g)
#     for g = 1:n_g
#         x[:,:, g] = deepcopy(image)
#     end

#     for i = 1:n_iterations
#         println("Iteration", i)
#         x = x + ddt_x(x, image, scales)
#     end
#     return x
# end

# # Oriented, elongated, spatially offset kernel G for Stage 3
# function oriented_kernel(i, j, p, q, k, scale)
#     m = sin.(pi*k/4)
#     n = cos.(pi*k/4)
#     return
# end

# # Shunting equation for Stage 3
# function ddt_y(x, y, alpha, scale)
#     n_row, n_column = size(x)
#     n_orientation = 4
#     dy = zeros(n_row, n_column, n_orientation)
#     for k = 1:n_orientation
#         for i = 1:n_row
#             for j = n_column
#                 dy[i,j,k] = -alpha*y[i,j,k] + (1-y[i,j,k])
#             end
#         end
#     end
# end

# # Stage 3: Contrast-sensitive oriented filtering
# function contrast_sensitive_oriented_filtering(image, x)
#     # Get the size of the field
#     n_row, n_column = size(x)

#     # Compute the LGN ON-cell and OFF-cell output signals
#     X_plus = [max(0, x[i,j]) for i=1:n_row, j=1:n_column]
#     X_minus = [max(0, -x[i,j]) for i=1:n_row, j=1:n_column]

#     return
# end

# # Stage 4: Contrast-insensitive oriented filtering
# function contrast_insensitive_oriented_filtering(image)
#     return
# end

# # Stage 5: Orientation competition at the same position
# function orientation_competition(image)
#     return
# end