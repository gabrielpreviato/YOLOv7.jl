using YOLOv7
using Flux
using Zygote

m = YOLOv7.load_model() |> gpu
x = rand(Float32, 640, 640, 3, 1)# |> gpu
m(x)

# # Expr[:(var"##312" = (layers[1])(x)), :(var"##313" = (layers[2])(var"##312")), :(var"##314" = (layers[3])(var"##313"))]

X = rand(Float32, 320, 320, 3, 10)
Y = rand([0, 1], 10)
data = Flux.DataLoader((X, Y), batchsize=1)
loss(x, y) = Flux.Losses.mse(m(x), y)
ps = Flux.params(m)
opt = Flux.Adam(0.001, (0.9, 0.8))

function my_custom_train!(loss, ps, data, opt)
    ps = Flux.Params(ps)
    for d in data
        d = d |> gpu
        # back is a method that computes the product of the gradient so far with its argument.
        train_loss, back = Zygote.pullback(() -> loss(d...), ps)
        # Insert whatever code you want here that needs training_loss, e.g. logging.
        # logging_callback(training_loss)
        # Apply back() to the correct type of 1.0 to get the gradient of loss.
        gs = back(one(train_loss))
        # Insert whatever code you want here that needs gradient.
        # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
        Flux.update!(opt, ps, gs)
        # Here you might like to check validation set accuracy, and break out to do early stopping.
    end
end

my_custom_train!(loss, ps, data, opt)
