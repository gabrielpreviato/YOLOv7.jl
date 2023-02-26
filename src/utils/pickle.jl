using Pickle
using Flux
using OrderedCollections

function get_pickle_conv(d::Dict{Any, Any})
    stride = d["stride"]
    pad = d["padding"]
    groups = d["groups"]
    dilation = d["dilation"]

    weight = convert(Array{Float32, 4}, d["_parameters"]["weight"].args[2])
    rev_weight = flip(permutedims(weight, ndims(weight):-1:1))
    bias = d["_parameters"]["bias"] === nothing ? false :  Float32.(d["_parameters"]["bias"].args[2])

    Flux.Conv(identity, rev_weight, bias, stride, pad, dilation, groups)
end

function get_pickle_bn(d::Dict{Any, Any})
    ϵ = Float32.(d["eps"])
    momentum = Float32.(d["momentum"])
    γ = Float32.(d["_parameters"]["weight"].args[2])
    β = Float32.(d["_parameters"]["bias"].args[2])
    μ = Float32.(d["_buffers"]["running_mean"])
    σ² = Float32.(d["_buffers"]["running_var"])
    chs = d["num_features"]

    Flux.BatchNorm(identity, β, γ, μ, σ², ϵ, momentum, true, true, nothing, chs)
end

function get_pickle_implicit(d::Dict{Any, Any})
    w = Float32.(d["_parameters"]["implicit"].args[2])
    permutedims(w, ndims(w):-1:1)
end

function get_pickle_mp(d::Dict{Any, Any})
    kernel = d["kernel_size"]
    pad = d["padding"]
    stride = d["stride"]

    Flux.MaxPool((kernel, kernel), (pad, pad), (stride, stride))
end

function get_pickle_sppcspc(d)
    cv1 = YOLOv7.Conv(
        get_pickle_conv(d.args[2]["_modules"]["cv1"].args[2]["_modules"]["conv"].args[2]),
        get_pickle_bn(d.args[2]["_modules"]["cv1"].args[2]["_modules"]["bn"].args[2]),
        YOLOv7.silu
    )

    cv2 = YOLOv7.Conv(
        get_pickle_conv(d.args[2]["_modules"]["cv2"].args[2]["_modules"]["conv"].args[2]),
        get_pickle_bn(d.args[2]["_modules"]["cv2"].args[2]["_modules"]["bn"].args[2]),
        YOLOv7.silu
    )

    cv3 = YOLOv7.Conv(
        get_pickle_conv(d.args[2]["_modules"]["cv3"].args[2]["_modules"]["conv"].args[2]),
        get_pickle_bn(d.args[2]["_modules"]["cv3"].args[2]["_modules"]["bn"].args[2]),
        YOLOv7.silu
    )

    cv4 = YOLOv7.Conv(
        get_pickle_conv(d.args[2]["_modules"]["cv4"].args[2]["_modules"]["conv"].args[2]),
        get_pickle_bn(d.args[2]["_modules"]["cv4"].args[2]["_modules"]["bn"].args[2]),
        YOLOv7.silu
    )

    cv5 = YOLOv7.Conv(
        get_pickle_conv(d.args[2]["_modules"]["cv5"].args[2]["_modules"]["conv"].args[2]),
        get_pickle_bn(d.args[2]["_modules"]["cv5"].args[2]["_modules"]["bn"].args[2]),
        YOLOv7.silu
    )

    cv6 = YOLOv7.Conv(
        get_pickle_conv(d.args[2]["_modules"]["cv6"].args[2]["_modules"]["conv"].args[2]),
        get_pickle_bn(d.args[2]["_modules"]["cv6"].args[2]["_modules"]["bn"].args[2]),
        YOLOv7.silu
    )

    cv7 = YOLOv7.Conv(
        get_pickle_conv(d.args[2]["_modules"]["cv7"].args[2]["_modules"]["conv"].args[2]),
        get_pickle_bn(d.args[2]["_modules"]["cv7"].args[2]["_modules"]["bn"].args[2]),
        YOLOv7.silu
    )

    m = [get_pickle_mp(mp.args[2]) for mp in values(d.args[2]["_modules"]["m"].args[2]["_modules"])]

    YOLOv7.SPPCSPC(cv1, cv2, cv3, cv4, m, cv5, cv6, cv7)
end

function load_pickle(pickle_path)
    pt = Pickle.Torch.THload(pickle_path)
    model = pt["model"].args[2]["_modules"]["model"].args[2]["_modules"]
    model_flux = OrderedDict()

    for (k, m) in model
        # println(m)
        if m.args[1].args[1].head == Symbol("models.common.Conv")
            # Get activation
            conv, bn, act = nothing, nothing, nothing
            if m.args[2]["_modules"]["act"].args[1].args[1].head == Symbol("torch.nn.modules.activation.SiLU")
                act = YOLOv7.silu
            else
                throw("Missing activation $(string(m.args[2]["_modules"]["act"].args[1].args[1].head)) definition")
            end
    
            # Get BatchNorm
            if m.args[2]["_modules"]["bn"].args[1].args[1].head == Symbol("torch.nn.modules.batchnorm.BatchNorm2d")
                bn = get_pickle_bn(m.args[2]["_modules"]["bn"].args[2])
            else
                throw("Missing BN $(string(m.args[2]["_modules"]["bn"].args[1].args[1].head)) definition")
            end
    
            # Get Conv
            if m.args[2]["_modules"]["conv"].args[1].args[1].head == Symbol("torch.nn.modules.conv.Conv2d")
                conv = get_pickle_conv(m.args[2]["_modules"]["conv"].args[2])
            else
                throw("Missing Conv $(string(m.args[2]["_modules"]["conv"].args[1].args[1].head)) definition")
            end
    
            model_flux[k] = YOLOv7.Conv(conv, bn, act)
        elseif m.args[1].args[1].head == Symbol("models.common.MP")
            mp = get_pickle_mp(m.args[2]["_modules"]["m"].args[2])
            model_flux[k] = mp
        elseif m.args[1].args[1].head == Symbol("models.common.Concat")
            continue
        elseif m.args[1].args[1].head == Symbol("models.common.SPPCSPC")
            sppcspc = get_pickle_sppcspc(m)
            model_flux[k] = sppcspc
        elseif m.args[1].args[1].head == Symbol("torch.nn.modules.upsampling.Upsample")
            scale_factor = m.args[2]["scale_factor"]
            mode = Symbol(m.args[2]["mode"])
    
            up = Flux.Upsample(Int(scale_factor), mode)
            model_flux[k] = up
        elseif m.args[1].args[1].head == Symbol("models.common.RepConv")
            act = nothing
            
            if m.args[2]["_modules"]["act"].args[1].args[1].head == Symbol("torch.nn.modules.activation.SiLU")
                act = YOLOv7.silu
            else
                throw("Missing activation $(string(m.args[2]["_modules"]["act"].args[1].args[1].head)) definition")
            end
    
            rbr_dense_conv = get_pickle_conv(m.args[2]["_modules"]["rbr_dense"].args[2]["_modules"]["0"].args[2])
            rbr_dense_bn = get_pickle_bn(m.args[2]["_modules"]["rbr_dense"].args[2]["_modules"]["1"].args[2])
    
            rbr_1x1_conv = get_pickle_conv(m.args[2]["_modules"]["rbr_1x1"].args[2]["_modules"]["0"].args[2])
            rbr_1x1_bn = get_pickle_bn(m.args[2]["_modules"]["rbr_1x1"].args[2]["_modules"]["1"].args[2])
    
            rep_conv = YOLOv7.RepConv(
                nothing,
                Flux.Chain(rbr_dense_conv, rbr_dense_bn),
                Flux.Chain(rbr_1x1_conv, rbr_1x1_bn),
                nothing,
                act
            )
            
            model_flux[k] = rep_conv
        elseif m.args[1].args[1].head == Symbol("models.yolo.IDetect")
            anchors = m.args[2]["_buffers"]["anchors"]
            anchor_grid = m.args[2]["_buffers"]["anchor_grid"]
    
            out_conv = [get_pickle_conv(d.args[2]) for d in values(m.args[2]["_modules"]["m"].args[2]["_modules"])]
            ia = [YOLOv7.ImplicitAddition(get_pickle_implicit(d.args[2])) for d in values(m.args[2]["_modules"]["ia"].args[2]["_modules"])]
            im = [YOLOv7.ImplicitMultiplication(get_pickle_implicit(d.args[2])) for d in values(m.args[2]["_modules"]["im"].args[2]["_modules"])]
    
            nc = m.args[2]["nc"]
            na = m.args[2]["na"]
            nl = m.args[2]["nl"]
            no = m.args[2]["no"]
    
            idetec = YOLOv7.IDetec(nc, no, nl, na, out_conv, ia, im, anchors, anchor_grid)
            model_flux[k] = idetec
        else
            throw("Missing Layer $(string(m.args[1].args[1].head)) definition")
        end
    end

    return model_flux, pt["model"].args[2]["names"], pt["model"].args[2]["nc"]
end