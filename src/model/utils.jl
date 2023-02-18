function get_pickle_conv(d::Dict{Any, Any})
    stride = d["stride"]
    pad = d["padding"]
    groups = d["groups"]
    dilation = d["dilation"]

    weight = Float32.(d["_parameters"]["weight"].args[2])
    rev_weight = flip(permutedims(weight, ndims(weight):-1:1))
    bias = Float32.(d["_parameters"]["bias"] === nothing ? false : d["_parameters"]["bias"].args[2])

    Flux.Conv(identity, rev_weight, bias, stride, pad, dilation, groups)
end

function get_pickle_bn(d::Dict{Any, Any})
    ϵ = d["eps"]
    momentum = d["momentum"]
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