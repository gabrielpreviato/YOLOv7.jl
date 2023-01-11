using YOLOv7
using Flux, ProgressMeter
using Zygote

model = YOLOv7.load_model(raw"src\conf\yolov7.yaml") |> gpu
# x = rand(Float32, 320, 320, 3, 1) |> gpu
# m(x)

# # Expr[:(var"##312" = (layers[1])(x)), :(var"##313" = (layers[2])(var"##312")), :(var"##314" = (layers[3])(var"##313"))]

X = rand(Float32, 320, 320, 3, 10)
Y = rand([0, 1], 10)
data = Flux.DataLoader((X, Y), batchsize=1)
loss(x, y) = Flux.Losses.mse(x, y)
# ps = Flux.params(m)
opt = Flux.setup(Flux.Adam(0.001, (0.9, 0.8)), model)

println("Created data.")

function my_custom_train!(loss, data, opt)
    @showprogress for epoch in 1:10
        # ps = Flux.Params(ps)
        for d in data
            println("Data Loop")
            input, label = d |> gpu
            # back is a method that computes the product of the gradient so far with its argument.
            val, grads = Flux.withgradient(model) do m
            # Insert whatever code you want here that needs training_loss, e.g. logging.
            # logging_callback(training_loss)
                result = m(input)
                loss(result, label)
            # Apply back() to the correct type of 1.0 to get the gradient of loss.
            end
            # Insert whatever code you want here that needs gradient.
            # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
            Flux.update!(opt, model, grads[1])
            # Here you might like to check validation set accuracy, and break out to do early stopping.
            println("a")
        end
    end
end

println("Starting training.")
my_custom_train!(loss, data, opt)
