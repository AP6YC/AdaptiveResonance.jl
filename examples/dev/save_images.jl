for i = 1:4
    for j = 1:4
        local_array = y[:,:,i,j]
        im_min = minimum(local_array)
        im_max = maximum(local_array)
        local_array = (local_array .- im_min) / (abs(im_min) + abs(im_max))
        save("filter"*string(i)*string(j)*".png", imresize(Gray.(local_array), ratio=4))
    end
end