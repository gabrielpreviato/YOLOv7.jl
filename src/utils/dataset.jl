using Images
struct ImageDataset
    files::Vector{String}
    fullpath_files::Vector{String}
end
ImageDataset(folder::String) = ImageDataset(readdir(folder), readdir(folder, join=true))

MLUtils.numobs(data::ImageDataset) = length(data.files)

function MLUtils.getobs(data::ImageDataset, idxs::Union{UnitRange{Int64}, Vector{Int64}})
    x = float.(Images.load.(data.fullpath_files[idxs]))
    x = imresize.(x, 320, 320)
    x = channelview.(x)
    x = permutedims.(x, ((3, 2, 1),))
    x = stack(x, dims=4)
    y = [data.files[i][1:3] == "cat" ? 1.0f0 : 0.0f0 for i in idxs]
    y = reshape(y, (1, length(idxs)))

    return (x, y)
end