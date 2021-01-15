
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