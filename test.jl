using YOLOv7
using Flux, ProgressMeter
using Zygote
using MLUtils
using DataLoaders
using ImageDraw
using ImageView
using Images

using CUDA
CUDA.allowscalar(false)

using BSON: @save, @load

mapping = Dict(
    "goal" => 1,
    "ball" => 2,
)
dataset = YOLOv7.ImageDataset(raw"C:\Users\gabri\Dropbox\IC\dataset\test6", mapping)

function my_custom_train!(loss, data, opt, model, BATCHSIZE, EPOCHS)
    for EPOCH in 1:EPOCHS
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
                l = loss(result, label)
                l
            # Apply back() to the correct type of 1.0 to get the gradient of loss.
            end
            val = ret.val[1]
            lbox, lobj, lcls, _ = ret.val[2]
            grads = ret.grad

            avg_loss += val
            # Insert whatever code you want here that needs gradient.
            # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
            Flux.update!(opt, model, grads[1])
            # Here you might like to check validation set accuracy, and break out to do early stopping.
            ProgressMeter.next!(p; showvalues = [(:iter,iter), (:avg_loss,avg_loss/iter),
                                                 (:loss,val), (:lbox,lbox), (:lobj, lobj), (:lcls,lcls)])
        end

        cpu_model = cpu(model)
        @save "mymodel_preload_$EPOCH.bson" cpu_model
    end
end

# anchors = Tuple([
#     [[12 16]; [19 36]; [40 28]],
#     [[36 75]; [76 55]; [72 146]],
#     [[142 110]; [192 243]; [459 401]]
# ] ./ [8, 16, 32])

anchors = Tuple([
    [[0.87646 0.76221]; [0.30786 3.21289]; [5.64453 0.39526]],
    [[0.27368 2.20898]; [0.39478 3.63867]; [4.15234 0.60596]],
    [[0.32495 2.86133]; [0.59619 4.11719]; [2.95312 1.13184]]
])

anchors_grid = anchors .* [16, 32, 64]

# anchors_grid =  Tuple([[6.0 8.0; 9.5 18.0; 20.0 14.0],
#                         [18.0 37.5; 38.0 27.5; 36.0 73.0],
#                         [71.0 55.0; 96.0 121.5; 229.5 200.5]])

hyper = Dict{String, Any}("cls_pw"=>1, "obj_pw"=>1, "anchor_t"=>8.0f0, "box"=>0.55f0, "cls"=>0.20f0, "obj"=>0.25f0)

function(m::Flux.Conv)(x::Dict)
    ret = m(x[:x])
    # Both original x and ret have a :x key
    # For the merge function, elements with the same key, 
    # the value for that key will be of the last Dict listed (ret)
    # return merge(x, ret)
    return ret
end

struct TestMerge
    mp
end

TestMerge() = TestMerge([
    MeanPool((4, 4)),
    MeanPool((2, 2)),
    identity,
])

function (m::TestMerge)(x::Vector{CuArray{Float32, 5, CUDA.Mem.DeviceBuffer}})
    cat(
        [
        m.mp[i](maximum(x_i; dims=1)[1, :, :, :, :])
        for (i, x_i) in enumerate(x)
        ]...;
        dims=3
    )
end


model = Chain(                                                                                                                                                                                  
    YOLOv7.YOLOv7Backbone(;p3=true, p4=true),
    YOLOv7.SPPCSPC(1024, 512),
    YOLOv7.YOLOv7HeadRouteback(512, :p4),
    YOLOv7.YOLOv7HeadBlock(256, :h1), # 63
    YOLOv7.YOLOv7HeadRouteback(256, :p3),
    YOLOv7.YOLOv7HeadBlock(128, :h2), # 75
    YOLOv7.YOLOv7HeadIncep(128, :h1),
    YOLOv7.YOLOv7HeadBlock(256, :h3), #88
    YOLOv7.YOLOv7HeadIncep(256, :sppcspc),
    YOLOv7.YOLOv7HeadBlock(512, :h4), #101
    YOLOv7.YOLOv7HeadTailRepConv(128, :h2, :h3, :h4),
    YOLOv7.IDetec(5; anchors=anchors, channels=(256, 512, 1024)),
) |> gpu

# Model warmup
x = rand(Float32, 320, 320, 3, 1) |> gpu
r = MLUtils.getobs(dataset, [1])
x, y =  r[1] |> gpu, r[2] |> gpu
ŷ = model(x)
# y = Float32.([
#     [1.00000e+00 1.00000e+00 4.98467e-01 5.35664e-01 7.84091e-01 9.28673e-01];
#     [1.00000e+00 4.0000e+00 4.76484e-01 5.27067e-01 1.92362e-01 1.85786e-01];
#     [1.00000e+00 3.0000e+00 4.12404e-01 2.75406e-01 1.51938e-01 1.38589e-01];
#     [1.00000e+00 3.0000e+00 6.79160e-01 8.22009e-01 5.55278e-02 3.55983e-01];
#     [1.00000e+00 4.0000e+00 6.54439e-01 4.39052e-01 1.68993e-01 2.62337e-01];
#     [1.00000e+00 2.0000e+00 4.88898e-01 7.81393e-01 3.31721e-01 3.29491e-01];
#     [1.00000e+00 4.0000e+00 4.02097e-01 1.44877e-01 2.20601e-01 1.42476e-01];
#     [1.00000e+00 5.0000e+00 4.84404e-01 7.71480e-01 2.39558e-01 2.30949e-01];
#     [1.00000e+00 5.0000e+00 6.36203e-01 2.95479e-01 9.84112e-02 1.23567e-01]
# ]') |> gpu

l = YOLOv7.loss(cl, ŷ, y, 1; nc=2)

opt_init = Flux.setup(Flux.Adam(0.0001), model[1:end-2])
opt_end = Flux.setup(Flux.Adam(0.001), model[end-2:end])
ret = Flux.withgradient(model) do m
# Insert whatever code you want here that needs training_loss, e.g. logging.
# logging_callback(training_loss)
    result = m(x)
    l = YOLOv7.loss(cl, result, y, 1; nc=2)
    # Flux.mae(result, [1.0] |> gpu)
    # Zygote.@showgrad(l)
# Apply back() to the correct type of 1.0 to get the gradient of loss.
end
grads = ret.grad
# Insert whatever code you want here that needs gradient.
# E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
Flux.update!(opt_init, model[1:end-2], grads[1])
Flux.update!(opt_end, model[end-2:end], grads[1])


mma = Chain(
    Conv((1,1), 1=>10),
    Conv((1,1), 10=>1),
    flatten,
    Dense(9, 1)
) |> gpu

opt_init = Flux.setup(Flux.Adam(0.0001), mma[1:2])
opt_end = Flux.setup(Flux.Adam(0.001), mma[3:end])

x = rand(Float32, 3, 3, 1, 1) |> gpu
ret = Flux.withgradient(mma) do m
# Insert whatever code you want here that needs training_loss, e.g. logging.
# logging_callback(training_loss)
    result = m(x)
    [sum(result) .- 1.0f0, 2, 1]
    # Flux.mae(result, [1.0] |> gpu)
    # Zygote.@showgrad(l)
# Apply back() to the correct type of 1.0 to get the gradient of loss.
end
grads = ret.grad
# Insert whatever code you want here that needs gradient.
# E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
Flux.update!(s, mma, grads[1])

# exit()
# # Expr[:(var"##312" = (layers[1])(x)), :(var"##313" = (layers[2])(var"##312")), :(var"##314" = (layers[3])(var"##313"))]

# X = rand(Float32, 320, 320, 3, 10)
# Y = rand([0, 1], 10)
# data = Flux.DataLoader((X, Y), batchsize=1)

# ps = Flux.params(m)
opt = Flux.setup(Flux.Adam(0.005), model)

# # Load cat-dog Kaggle dataset
# dataset = ImageDataset(raw"C:\Users\gabri\Downloads\train\train")

BATCHSIZE = 1
EPOCHS = 20
# data = DataLoaders.DataLoader(dataset, BATCHSIZE)
data = MLUtils.DataLoader(dataset, batchsize=BATCHSIZE, shuffle=true)


# loss(ŷ, y) = Flux.mse(ŷ, y)
println("Created data.")

println("Starting training.")

@load "mymodel_preload_4.bson" cpu_model

model = cpu_model |> gpu
# m = YOLOv7.yolov7(pretrained=true)
# model = m.m |> gpu

opt_backbone = Flux.setup(Flux.Adam(0.00005), model[1:2])
opt_middle = Flux.setup(Flux.Adam(0.0001), model[3:end-2])
opt_detec = Flux.setup(Flux.Adam(0.001), model[end-1:end])

opt = (layers=(opt_backbone.layers..., opt_middle.layers..., opt_detec.layers...),)

cl = YOLOv7.ComputeLoss(hyper, model)
loss(ŷ, y) = YOLOv7.loss(cl, ŷ, y, BATCHSIZE; nc=2)

my_custom_train!(loss, data, opt, model, BATCHSIZE, EPOCHS)

function _make_grid(nx=20, ny=20)
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
end

make_grid(x, y) = [x y]

function output_to_box(ŷ, anchors_grid, stride)
    z = []
    

    for (i, ŷ_i) in enumerate(ŷ)
        xs, ys = size(ŷ_i)[2:3]
        bs = size(ŷ_i)[end]
        no = size(ŷ_i)[1]
        
        grid = reshape(
            stack([((1:xs)' .* ones(ys))', ((1:ys)' .* ones(xs))]; dims=1),
            (2, ys, xs, 1, 1))
        # println(grid)
        # grid = repeat(grid, 1, 1, 1, 3, 1)
        # grid = reshape((1:xs)' .* ones(ys), (1, ys, xs, 1, 1))

        sig = sigmoid(ŷ_i) |> cpu
        
        # println(size(sig[1:2, :, :, :, :]))
        # println(size(grid))

        println(grid[1:2, 1:2, 1:2, :, :])
        println(sig[1:2, 1:2, 1:2, :, :])
        println(stride[i])

        sig[1:2, :, :, :, :] .= @. (sig[1:2, :, :, :, :] * 2.0 - 0.5 + grid) * stride[i]

        println(sig[1:2, 1:2, 1:2, :, :])
        println("-----")

        # println(size(sig[3:4, :, :, :, :]))
        # println(size(reshape(anchors_grid[i]', (2, 1, 1, 3, 1))))

        anch_g = reshape(anchors_grid[i]', (2, 1, 1, 3, 1))
        sig[3:4, :, :, :, :] .= @. (sig[3:4, :, :, :, :] * 2.0) ^ 2 
        sig[3:4, :, :, :, :] .= sig[3:4, :, :, :, :] .* anch_g

        # push!(z, sig[:, :, :, :, :])
        push!(z, reshape(sig, (size(sig)[1], :, size(sig)[end])))
    end
        # if self.grid[i].shape[2:4] != x[i].shape[2:4]:
        #     self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

        # y = x[i].sigmoid()
        # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
        # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
        # z.append(y.view(bs, -1, self.no))
    
    hcat(z...)
end

function xywh2xyxy(x)
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = copy(x)
    y[1, :] .= @. x[1, :] - x[3, :] / 2  # top left x
    y[2, :] .= @. x[2, :] - x[4, :] / 2  # top left y
    y[3, :] .= @. x[1, :] + x[3, :] / 2  # bottom right x
    y[4, :] .= @. x[2, :] + x[4, :] / 2  # bottom right y
    return y
end

function nms(boxes, scores, iou_thres)
    detec::Array{Int} = []
    boxes_cp = []
    scores_cp = copy(scores)
    for i in 1:size(boxes)[2]
        push!(boxes_cp, boxes[:, i])
    end

    # println(boxes_cp)

    while sum(scores_cp) > 0
        i = argmax(scores_cp)
        # println(scores_cp)
        # println(i)

        add = true
        for d in detec
            iou = YOLOv7.bbox_iou(boxes[:, d], boxes[:, i])[1]
            # println(iou)
            if iou > iou_thres
                add = false
                break
            end
        end

        if add
            push!(detec, i)
        end

        scores_cp[i] = 0
    end

    # println(detec)
    # detec = stack(detec, dims=1)
    # println(size(detec))

    return detec
end

function non_max_suppression(prediction; nc=1, conf_thres=0.25, iou_thres=0.45, classes=nothing, agnostic=false, multi_label=false, labels=())
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = true  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = false  # use merge-NMS

    output = repeat([Flux.zeros32(0, 6)], size(prediction)[1])
    println(size(output))
    for (i, x) in enumerate(prediction)  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        nc = size(x)[1] - 5  # number of classes
        xc = x[5, :, :] .> conf_thres  # candidates
        x = x[:, :, :]

        x = permutedims(x, ((2:ndims(x) |> collect)..., 1))

        x = x[xc, :]'  # confidence

        println(x)

        println(size(x))

        # Cat apriori labels if autolabelling
        # if length(labels) != 0 && length(labels[i]) > 0
        #     l = labels[i]
        #     v = Flux.zeros32(nc + 5, size(l)[2])
        #     println(size(v), size(l))
        #     println(size(v[1:4, :]), size(l[2:5, :]))
        #     v[1:4, :] .= l[2:5, :]  # box
        #     v[5, :] .= 1.0  # conf

        #     println(Int.(l[1, :]))
        #     v[Int.(l[1, :]) .+ 5, 1:size(l)[2]] .= 1.0  # cls

        #     println(size(x), size(v))

        #     x = cat(x, v, dims=2)
        # end


        # If none remain process next image
        if size(x)[1] == 0
            continue
        end

        # Compute conf
        if nc == 1
            x[6:end, :] = x[5, :] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else
            println(size(x[6:end, :]), size(x[5, :]))
            x[6:end, :] .= x[6:end, :] .* x[5, :]'  # conf = obj_conf * cls_conf
        end

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[1:4, :])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label
            i, j = (x[:, 5:end] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else  # best class only

            # conf, j = x[:, 5:end].max(1, keepdim=True)
            conf, j = findmax(x[6:end, :], dims=1)

            println(size(box))
            println(size(conf))
            println(size(map(x -> float.(x[1]), j)))

            x = vcat(box, conf, map(x -> float.(x[1]), j))'
            x = x[conf[1, :] .> conf_thres, :]'
        end

        println(size(x))

        # Filter by class
        if classes !== nothing
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        end

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = size(x)[2]  # number of boxes
        if n == 0  # no boxes
            continue
        elseif n > max_nms  # excess boxes
            x = x[sortperm(x[5, :])[1:max_nms]]  # sort by confidence
        end

        # Batched NMS
        c = x[6, :] .* (agnostic ? 0 : max_wh)  # classes]

        boxes = x[1:4, :] .+ c'   # boxes (offset by class) 
        scores = x[5, :]        # scores
        i_nms = nms(boxes, scores, iou_thres)  # NMS
        if length(i_nms) > max_det  # limit detections
            i_nms = i_nms[1:max_det]
        end
        if merge && (1 < n < 3E3)  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i_nms, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant
                i_nms = i_nms[iou.sum(1) > 1]
            end
        end  # require redundancy

        output[i] = x[:, i_nms]
    end

    return output
end


# box1 = [[ 5.39401e-01  8.82256e-01  9.31658e-01  5.36098e-01  1.87861e-01  3.43929e-01 -4.89734e-01  2.57725e-01  1.15654e-01  5.45307e-01  5.14027e-01  4.82989e-01 -2.98187e-01  9.74460e-02 -2.53039e-01  2.63856e-01  7.24840e-01  2.47019e-01  5.99235e-01  8.87640e-01  1.10212e+00  3.96790e-01 5.31852e-01 -2.74390e-01  7.03425e-01  3.26048e-01 -3.57962e-01  4.50286e-01  2.10744e-01  1.17221e+00  7.57649e-01  3.07920e-01  5.39139e-01  5.16832e-01 -4.18562e-01  9.56085e-01  4.44160e-01  6.36602e-01  2.32696e-01  5.29838e-01  5.11759e-01  6.05975e-01  5.15997e-01  7.76816e-01 1.19587e+00  5.99152e-01  2.58974e-01 -2.23276e-01  2.90447e-01  5.41702e-01  3.47662e-01  1.36219e+00 -2.70276e-01  5.72045e-01];
#         [ 1.80391e-01 -2.43786e-01 -1.06426e-02  5.20606e-01  1.19105e+00  1.44914e-01 -7.18905e-02 -2.15239e-01 -2.13494e-01  4.76015e-01  5.16348e-01  4.50635e-01  4.07135e-02  3.49769e-01  9.10657e-01  1.09658e+00  8.53297e-01  4.05993e-01  2.97365e-01  2.42934e-01  5.82251e-01 -8.20818e-02 4.44801e-01 -4.95489e-02 -1.98249e-01  6.76655e-01 -1.91582e-01 -3.77717e-01  2.02549e-01 -1.64254e-01 -4.67674e-01 -3.29197e-01  5.29436e-01  5.24202e-01 -3.31250e-01 -2.60239e-01  3.62784e-01  5.06191e-01  7.44006e-01  5.31759e-01  5.21847e-01  4.67220e-01  5.79118e-01 -2.46831e-01 3.55352e-01  5.11442e-01 -2.04771e-02  1.25489e-01  4.50468e-02  4.68938e-01  3.05509e-01  1.27490e+00  1.31057e+00  9.05805e-02];
#         [ 2.78292e+00  1.15062e+00  1.02049e+00  2.83127e+00  4.01758e-02  9.98696e-01  7.57095e-01  3.09071e+00  1.73243e+00  5.76003e+00  5.60353e+00  4.66936e+00  5.37023e+00  4.41759e+00  1.71151e+01  1.74152e+01  6.07015e+00  1.21251e+01  1.10842e+00  1.13561e+00  2.38489e-01  3.52790e+00 1.90346e+00  4.45808e+00  8.10387e+00  4.82766e+00  1.74144e-01  5.66704e+00  7.16238e+00  2.47512e-01  4.89044e-03  7.86683e-01  5.45711e+00  5.32865e+00  2.22200e+00  2.60491e+00  2.56581e+00  3.05898e+00  4.08502e+00  5.51527e+00  5.83542e+00  5.85163e+00  4.85549e+00  1.47472e+00 6.28971e-01  2.64628e+00  1.76301e+00  1.08816e+00  4.49436e+00  6.07975e+00  5.71914e+00  1.55464e+01  1.01647e+01  4.31287e+00];
#         [ 1.39994e+00  2.32690e+00  1.18626e+00  3.29443e+00  1.46395e+01  8.13055e+00  1.68319e+01  1.34682e+01  9.42356e+00  4.01659e+00  3.81802e+00  3.88898e+00  3.64387e-01  6.73184e+00  1.06770e+01  1.49551e+00  3.76953e+00  5.42964e+00  2.38911e+00  2.64251e+00  1.95322e+00  7.65295e+00 7.43398e+00  1.12060e+01  9.11563e+00  7.80275e+00  3.15336e-04  2.93101e-02  4.99744e+00  6.35535e+00  1.22363e-01  1.45145e+00  3.94138e+00  3.94876e+00  7.15332e+00  9.49909e+00  5.28921e+00  3.66330e+00  4.20075e+00  4.12877e+00  3.73481e+00  3.47150e+00  5.24752e+00  5.42893e-01 9.95821e-01  4.00071e+00  8.82310e+00  1.65477e+01  7.67616e+00  3.79397e+00  5.63849e+00  1.51163e+00  1.30462e-01  4.55712e+00]]

# box2 = [[ 0.25000  0.43750  0.62500  0.94991  0.12500  0.25000  0.25000  0.43750  0.62500  0.71368  0.53175  0.94991  0.12500  0.25000  0.25000  0.18750  0.43750  0.62500  1.25000  1.43750  1.12500  1.25000  1.25000  1.43750  1.12500  1.25000  1.25000  1.18750  1.43750  0.43750  0.12500 0.43750  0.71368  0.53175  0.12500  0.43750 -0.37500 -0.05009 -0.37500 -0.28632 -0.46825 -0.05009 -0.37500  0.25000  0.62500  0.94991  0.25000  0.25000  0.62500  0.94991  0.25000  0.25000  0.18750  0.62500];
#         [ 0.75000  0.18750  0.68750  0.76989  0.25000  0.75000  0.56250  0.18750  0.68750  0.17662  0.23527  0.76989  0.25000  0.75000  0.56250  0.75000  0.18750  0.68750  0.75000  0.18750  0.25000  0.75000  0.56250  0.18750  0.25000  0.75000  0.56250  0.75000  0.18750  1.18750  1.25000 1.18750  1.17662  1.23527  1.25000  1.18750  0.68750  0.76989  0.68750  0.17662  0.23527  0.76989  0.68750 -0.25000 -0.31250 -0.23011 -0.25000 -0.43750 -0.31250 -0.23011 -0.25000 -0.43750 -0.25000 -0.31250];
#         [ 3.50000  3.37500  5.00000  7.81126  7.00000  3.50000  7.50000  3.37500  5.00000 12.05988 17.50991  7.81126  7.00000  3.50000  7.50000 11.37500  3.37500  5.00000  3.50000  3.37500  7.00000  3.50000  7.50000  3.37500  7.00000  3.50000  7.50000 11.37500  3.37500  3.37500  7.00000 3.37500 12.05988 17.50991  7.00000  3.37500  5.00000  7.81126  5.00000 12.05988 17.50991  7.81126  5.00000  3.50000  5.00000  7.81126  3.50000  7.50000  5.00000  7.81126  3.50000  7.50000 11.37500  5.00000];
#         [ 5.50000  3.12500  3.12500  9.80796  7.00000  5.50000  7.12500  3.12500  3.12500 11.00032  4.47053  9.80796  7.00000  5.50000  7.12500 11.25000  3.12500  3.12500  5.50000  3.12500  7.00000  5.50000  7.12500  3.12500  7.00000  5.50000  7.12500 11.25000  3.12500  3.12500  7.00000 3.12500 11.00032  4.47053  7.00000  3.12500  3.12500  9.80796  3.12500 11.00032  4.47053  9.80796  3.12500  5.50000  3.12500  9.80796  5.50000  7.12500  3.12500  9.80796  5.50000  7.12500 11.25000  3.12500]]

# iou = YOLOv7.bbox_iou(box1, box2; x1y1x2y2=false, CIoU=true)

# iou_true = [ 1.77741e-01  2.27882e-01  5.93317e-02  1.20263e-01 -4.72749e-02  2.38556e-01  3.67112e-02  1.99946e-01  1.21980e-01  1.73895e-01  2.69958e-01  2.34626e-01  3.85087e-04  6.44479e-01  2.91131e-01  8.82612e-02  3.80344e-01  2.36002e-01  1.22723e-01  2.63714e-01 -1.89192e-02  4.44494e-01 2.27799e-01  1.02728e-01  6.60760e-01  4.45840e-01 -7.34454e-02 -5.34325e-02  2.74842e-01 -3.03111e-02 -7.55774e-02 -3.67094e-02  1.60411e-01  2.25073e-01  1.98760e-01  2.38484e-01  3.42293e-01  1.42826e-01  5.90409e-01  1.68657e-01  2.72987e-01  2.58221e-01  4.28273e-01  8.90735e-03 1.14812e-02  1.33897e-01  3.81022e-01  7.41449e-02  3.64226e-01  2.93955e-01  5.11514e-01  1.27025e-01 -4.62689e-02  6.12905e-01]

# function nearly_equal(a, b; eps=1e-4)
#     diff = abs.(a .- b)
#     return diff .< eps
# end

# equal = nearly_equal(iou, iou_true')
# println("Equal: ", sum(equal) == size(equal)[1])

# a = rand(Float32, 40, 40, 256, 2)
# b = rand(Float32, 1, 1, 256, 1)
strid = [8, 16, 32]

r = MLUtils.getobs(dataset, [2500])
x, y =  r[1] |> gpu, r[2] |> gpu
ŷ = model(x)
l = YOLOv7.loss(cl, ŷ, y, 1; nc=2)

img = copy(x[:, :, :, 1]) |> cpu
img_CHW = permutedims(img, (3, 2, 1))
img_rgb = colorview(RGB, img_CHW)

out = output_to_box(ŷ, anchors_grid, strid)
out_nms = non_max_suppression([out], nc=5)[1]
colors = [RGB{Float32}(1.0, 0.0, 0.0), RGB{Float32}(0.0, 1.0, 0.0), RGB{Float32}(0.0, 0.0, 1.0)]

conf1 = out[5, :, :] .> 0.25
d1 = permutedims(out, (2, 3, 1))[conf1, :]
d1 = trunc.(Int, d1)'
for i in 1:size(d1)[2]
    x1, y1 = d1[1, i] - d1[3, i] ÷ 2, d1[2, i] - d1[4, i] ÷ 2
    x2, y2 = d1[1, i] + d1[3, i] ÷ 2, d1[2, i] + d1[4, i] ÷ 2
    draw!(img_rgb, Polygon(RectanglePoints(x1, y1, x2, y2)), colors[1])
end

out_nms_trunc = trunc.(Int, out_nms)
for i in 1:size(out_nms_trunc)[2]
    x1, y1 = out_nms_trunc[1, i], out_nms_trunc[2, i]
    x2, y2 = out_nms_trunc[3, i], out_nms_trunc[4, 1]
    draw!(img_rgb, Polygon(RectanglePoints(x1, y1, x2, y2)), colors[2])
end

imshow(img_rgb)

function build_targets(compute_loss::YOLOv7.ComputeLoss, ŷ, y, x; device=gpu)
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

    na, nt = compute_loss.na, size(y)[2]  # number of anchors, targets
    # tcls, tbox, indices, anch = [], [], [], []
    
    gain = Flux.ones32(7) |> device # normalized to gridspace gain
    ai = repeat(range(1, na), 1, nt) |> device
    ai = permutedims(ai, (2, 1))

    targets = cat(repeat(y, 1, 1, na), reshape(ai, (1, size(ai)...)), dims=1) |> device # append anchor indices

    # println(targets[:, :, 1])

    # final targets = (7, 4, 3)
    g = 0.5  # bias
    off = [[0  0];
           [1  0]; [0  1]; [-1  0]; [0  -1]  # j,k,l,m
         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ]' .* g  # offsets
    
    # Zygote.@showgrad(off)
    
    ret = Zygote.@ignore [
        let
            anchors = compute_loss.anchors[i] |> device

            # Zygote.@showgrad(anchors)
            # println("size(anchors) ", size(anchors))
            # println("anchors ", anchors)
            # println("size(anchors') ", size(anchors'))
            # println("anchors' ", anchors')

            # println(size(ŷ[i]))
            # println(size(ŷ[i])[[2, 3, 2, 3]])

            # println("before gain ", size(gain), typeof(gain), gain)
            
            # new_gain = Zygote.Buffer(gain)
            # new_gain = gain
            # new_gain[3:6] = size(ŷ[i])[[2, 3, 2, 3]]  # xyxy gain
            # gain = copy(new_gain)

            gain = CUDA.@allowscalar [gain[1:2]..., size(ŷ[i])[[2, 3, 2, 3]]..., gain[7]] |> device

            # println("gain ", size(gain), typeof(gain), gain)

            # Match targets to anchors
            # println("size(targets) ", size(targets))
            # println("size(gain) ", size(gain))
            t = targets .* gain

            # Zygote.@showgrad(t)
            # println("size(t) ", size(t))
            # println("t1", t[:, :, 1])

            if nt != 0
                # Matches
                r = t[5:6, :, :] ./ reshape(anchors', 2, 1, 3)  # wh ratio
                # println("rsize ", size(r))
                
                j = max.(r, 1.f0 ./ r)
    
                # println("jsize before filter ", size(j))
                # println(j)

                
                j = maximum(j, dims=1)[1, :, :]

                # print("j to be compared ", j)

                j = j .< compute_loss.hyper["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # println("jsize after filter ", size(j))
                
                # println("tsize before filter ", size(t))
                t = t[:, j]  # filter
                # println("t1 after filter", t[:, :, 1])

                # println("tsize after filter ", size(t))
                # println("t ", t)
                if size(t)[end] != 0
                    # Offsets
                    gxy = t[3:4, :]  # grid xy
                    # println("gxy ", gxy)
                    # println("gain[[3, 4]] ", size(gain[[3, 4]]))
                    gxi = gain[[3, 4]] .- gxy  # inverse
                    # println("gxi ", size(gxi))

                    # println("gxi: ", gxi)
                    # println("typeof(gxy) ", typeof(gxy))

                    # j, k = @. ((gxy % 1.0f0 < g) & (gxy > 1.0f0))
                    jk = @. ((gxy % 1.0f0 < g) & (gxy > 1.0f0))
                    j = jk[1, :]
                    k = jk[2, :]

                    # println("jk ", jk)
                    # println("k ", size(k))

                    lm = @. ((gxi % 1.0f0 < g) & (gxi > 1.0f0))
                    l = lm[1, :]
                    m = lm[2, :]

                    # println("lm ", lm)

                    # println("l ", size(l))
                    # println("m ", size(m))


                    j = Bool.(cat(Flux.ones32(size(j)...) |> device, j, k, l, m; dims=2))
                    # println("size(j) ", size(j))
                    # println("j ", j)


                    t = repeat(t, 1, 1, 5)
                    # println("size(t) ", size(t))
                    # println("typeof(t) ", typeof(t))
                    t = t[:, j]
                    # println("t: ", t[:, :])
                    # println("size(t) ", size(t))
                    # println("typeof(t) ", typeof(t))

                    # println("size(off) ", size(off))
                    # println("size(gxy) ", size(gxy))

                    # println(size(Flux.zeros32(size(gxy)..., size(off)[2])))
                    # println(size(repeat(off, 1, 1, size(gxy)[2])))
                    perm_off = permutedims(repeat(off, 1, 1, size(gxy)[2]), (1, 3, 2)) |> device
                    # println(size(perm_off))

                    offsets = (zeros(size(gxy)..., size(off)[2]) |> device) .+ perm_off
                    # println("size(offsets) ", size(offsets))
                    offsets = offsets[:, j]
                    # println("size(offsets) ", size(offsets))
                else
                    t = targets[:, 1, :]
                    # println(size(targets))
                    offsets = 0
                end
            else
                t = targets[:, 1, :]
                # println(size(targets))
                offsets = 0
            end
            
            # Define
            # println("size(t) ", size(t))
            b = Int.(trunc.(t[1, :]))
            c = Int.(trunc.(t[2, :]))  # image, class
            gxy = t[3:4, :]  # grid xy
            gwh = t[5:6, :]  # grid wh

            # println(size(t))
            

            gij = Int.(trunc.(gxy .- offsets))
            gi = gij[1, :]  # grid xy indices
            gj = gij[2, :]

            # Append
            a = Int.(trunc.(t[7, :]))  # anchor indices
            # println(a)
            
            ix = CUDA.@allowscalar Int.(min.(max.(gi, 1), gain[3] |> cpu))
            ij = CUDA.@allowscalar Int.(min.(max.(gj, 1), gain[4] |> cpu))

            indices = CUDA.@allowscalar [CartesianIndex(i, j, k, l) for (i, j, k, l) in zip(
                ix,
                ij,
                a,
                b)]

            # tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # println("size(gwh) ", size(gwh))
            # println("size(gxy - gij) ", size(gxy - gij))
            # push!(tbox, cat(gxy - gij, gwh, dims=1))


            # anch.append(anchors[a])  # anchors
            # println("size(anchors) ", size(anchors))
            # println("size(a) ", size(a))
            anch = CUDA.@allowscalar anchors[[Int.(a)]...]

            # tcls.append(c)  # class
            # push!(tcls, c)
            # tcls, tbox, indices, anch
            pp = cat(gxy, gwh, dims=1)

            # Zygote.@showgrad(pp)

            [c, pp, indices, anch]
        end
        for i in 1:compute_loss.nl
    ]

    tcls = [i[1] for i in ret]
    # println(size.(tcls), typeof(tcls))
    # println(size(tcls[1]), typeof(tcls[1]))

    tbox = [i[2] for i in ret]
    indices = [i[3] for i in ret]
    anch = [i[4] for i in ret]

    return tcls, tbox, indices, anch
end

tcls, tbox, indices, anchorss = build_targets(cl, ŷ, y, x)

corrected_box = [tbox[1].*8 tbox[2].*16 tbox[3].*32]

img = copy(x[:, :, :, 1]) |> cpu
img_CHW = permutedims(img, (3, 2, 1))
img_rgb = colorview(RGB, img_CHW)

d1 = corrected_box |> cpu
d1 = Int.(trunc.(d1))
for i in 1:size(d1)[2]
    x1, y1 = d1[1, i] - d1[3, i] ÷ 2, d1[2, i] - d1[4, i] ÷ 2
    x2, y2 = d1[1, i] + d1[3, i] ÷ 2, d1[2, i] + d1[4, i] ÷ 2
    draw!(img_rgb, Polygon(RectanglePoints(x1, y1, x2, y2)), RGB{Float32}(1.0, 1.0, 0.0))
end

imshow(img_rgb)