using YOLOv7
using Flux

m = YOLOv7.load_model()
x = rand(Float32, 640, 640, 3, 1)
show(m(x))

X = rand(Float32, 640, 640, 3, 10)
Y = rand(0:9, 100)
data = Flux.DataLoader((X, Y), batchsize=1) 
loss(x, y) = Flux.Losses.mse(m(x), y)
ps = Flux.params(m)
opt = Flux.Adam(0.001, (0.9, 0.8))

Flux.train!(loss, ps, data, opt)