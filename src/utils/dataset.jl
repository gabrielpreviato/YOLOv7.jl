using Images
using MLUtils
using JSON
using Flux
using DataLoaders

struct ImageDataset
    mapping::Dict{String, Int}
    dataset_path::String
    image_files::Vector{String}
    label_files::Vector{String}
    
end

function ImageDataset(folder::String, mapping::Dict{String, Int})
    image_files = filter(x -> split(x, ".")[end] in ["jpg", "jpeg", "png"],
        readdir(folder * "\\images"; join=true)
        )
    label_files = filter(x -> split(x, ".")[end] in ["json"],
        readdir(folder * "\\obstacles"; join=true)
    )

    ImageDataset(mapping, folder, image_files, label_files)
end

MLUtils.numobs(data::ImageDataset) = length(data.image_files)

function load_label(data::ImageDataset, idxs::Union{UnitRange{Int64}, Vector{Int64}})
    labels = Array{Any, 2}(undef, 6, 0)

    for (i, label_file) in enumerate(data.label_files[idxs])
        jf = JSON.parsefile(label_file)

        cell_size = jf["cell_size"]
        img_width, img_height = jf["image_width"], jf["image_height"]
   
        for (j, obj) in enumerate(jf["objects"])
            xc = (obj["x_cell"] + obj["x_cell_position"]) * cell_size / img_width
            yc = (obj["y_cell"] + obj["y_cell_position"]) * cell_size / img_height
            w = obj["width"] / img_width
            h = obj["height"] / img_height
            c = data.mapping[obj["label"]]

            labels = hcat(labels, [i, c, xc, yc, w, h])
        end
    end

    return labels
end

# function load_label(data::ImageDataset, idx::Int64)
#     labels = Array{Any, 2}(undef, 6, 0)

#     label_file = data.label_files[idx]
#     jf = JSON.parsefile(label_file)

#     cell_size = jf["cell_size"]
#     img_width, img_height = jf["image_width"], jf["image_height"]

#     for (j, obj) in enumerate(jf["objects"])
#         xc = (obj["x_cell"] + obj["x_cell_position"]) * cell_size / img_width
#         yc = (obj["y_cell"] + obj["y_cell_position"]) * cell_size / img_height
#         w = obj["width"] / img_width
#         h = obj["height"] / img_height
#         c = data.mapping[obj["label"]]

#         labels = hcat(labels, [c, xc, yc, w, h])
#     end

#     return labels
# end

function MLUtils.getobs(data::ImageDataset, idxs::Union{UnitRange{Int64}, Vector{Int64}})
    x = float.(Images.load.(data.image_files[idxs]))
    x = imresize.(x, 320, 320)
    x = channelview.(x)
    x = permutedims.(x, ((3, 2, 1),))
    x = stack(x, dims=4)


    y = float.(load_label(data, idxs))
    # x = rand(Float32, 320, 320, 3, 1)
    # y = rand([0.0f0, 1.0f0], length(idxs))

    return (x, y)
end
