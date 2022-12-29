using YOLOv7
using Flux, ProgressMeter
using Zygote
using MLUtils
using CUDA

using Images
struct ImageDataset
    files::Vector{String}
    fullpath_files::Vector{String}
end
ImageDataset(folder::String) = ImageDataset(readdir(folder), readdir(folder, join=true))

MLUtils.numobs(data::ImageDataset) = length(data.files)

function MLUtils.getobs(data::ImageDataset, idxs::Union{UnitRange{Int64}, Vector{Int64}})
    x = float.(Images.load.(data.fullpath_files[idxs]))
    x = imresize.(x, 160, 160)
    x = channelview.(x)
    x = permutedims.(x, ((3, 2, 1),))
    x = stack(x, dims=4)
    y = [data.files[i][1:3] == "cat" ? 1.0f0 : 0.0f0 for i in idxs]
    y = reshape(y, (1, length(idxs)))

    return (x, y)
end

function my_custom_train!(loss, data, opt, model, BATCHSIZE)
    for EPOCH in 1:5
        # ps = Flux.Params(ps)
        p = Progress(length(data) ÷ BATCHSIZE, dt=1.0)
        avg_loss = 0.0
        for (iter, d) in enumerate(data)
            input, label = d |> gpu
            # back is a method that computes the product of the gradient so far with its argument.
            ret = Flux.withgradient(model) do m
            # Insert whatever code you want here that needs training_loss, e.g. logging.
            # logging_callback(training_loss)
                result = m(input)
                loss(result, label)
            # Apply back() to the correct type of 1.0 to get the gradient of loss.
            end
            val = ret.val
            grads = ret.grad

            avg_loss += val
            # Insert whatever code you want here that needs gradient.
            # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
            Flux.update!(opt, model, grads[1])
            # Here you might like to check validation set accuracy, and break out to do early stopping.
            ProgressMeter.next!(p; showvalues = [(:iter,iter), (:avg_loss,avg_loss/iter), (:loss,val)])
        end
    end
end

let 
    model = YOLOv7.load_model(raw"src\conf\yolov7.yaml") |> gpu
    # x = rand(Float32, 160, 160, 3, 1) |> gpu
    # model(x)

    # exit()
    # # Expr[:(var"##312" = (layers[1])(x)), :(var"##313" = (layers[2])(var"##312")), :(var"##314" = (layers[3])(var"##313"))]

    # X = rand(Float32, 320, 320, 3, 10)
    # Y = rand([0, 1], 10)
    # data = Flux.DataLoader((X, Y), batchsize=1)
    loss(x, y) = sqrt(Flux.Losses.mse(x, y))
    # ps = Flux.params(m)
    opt = Flux.setup(Flux.Adam(0.001), model)

    # Load cat-dog Kaggle dataset




    dataset = ImageDataset(raw"C:\Users\gabri\Downloads\train\train")

    BATCHSIZE = 4
    data = Flux.DataLoader(dataset, batchsize=BATCHSIZE, shuffle=true)

    println("Created data.")

    println("Starting training.")
    my_custom_train!(loss, data, opt, model, BATCHSIZE)
end