using Flux
using Zygote
using CUDA
using Distributions, Random
using OrderedCollections: OrderedDict
using LinearAlgebra

using YOLOv7: Node

ϵ = 0.00001f0
momentum = 0.970f0

flip(x) = x[end:-1:1, end:-1:1, :, :]

struct Conv
    conv::Flux.Conv
    bn::Union{Flux.BatchNorm, typeof(identity)}
    act::Function
end

Conv(conv::Flux.Conv, bn::Flux.BatchNorm) = Conv(conv, bn, silu)
Conv(conv::Flux.Conv, bn::typeof(identity)) = Conv(conv, bn, silu)

function Conv(c::Pair{Int64, Int64}, kernel, stride)
    return Conv(
        Flux.Conv((kernel, kernel), c; stride=stride, pad=Flux.SamePad()),
        Flux.BatchNorm(c.second)
    )
end

Flux.@functor Conv

function fuse(conv::Flux.Conv, bn::Flux.BatchNorm)
    w_conv = conv.weight
    w_bn = reshape(bn.γ ./ sqrt.(bn.ϵ .+ bn.σ²), (1, 1, 1, :))

    w_fusedconv = w_conv .* w_bn

    b_conv = conv.bias == false ? zeros32(size(w_conv)[4]) : conv.bias
    b_bn = bn.β .- (bn.γ.*bn.μ./sqrt.(bn.ϵ .+ bn.σ²))

    b_fusedconv = w_bn[1, 1, 1, :] .* b_conv + b_bn

    return Flux.Conv(w_fusedconv, b_fusedconv; stride=conv.stride, pad=conv.pad, dilation=conv.dilation)
end

function fuse(m::Conv)
    if typeof(m.bn) == Flux.BatchNorm
        return Conv(
            fuse(m.conv, m.bn),
            identity,
            m.act
        )
    else
        return m
    end
end

function (m::Conv)(x::AbstractArray)
    # println(x)
    m.act(m.bn(m.conv(x)))
    # m.act(m.conv(x))
end

function (m::Conv)(x::Dict)
    # println(x)
    x[:x] = m.act(m.bn(m.conv(x[:x])))
    return x
end

Base.show(io::IO, obj::Conv) = show(io, "$(obj.conv); $(obj.bn); $(obj.act)")

# function (m::Conv)(x::Symbol)
#     # println(x)
#     return x
# end

struct SPPCSPC
    cv1::Conv
    cv2::Conv
    cv3::Conv
    cv4::Conv
    m::Array{Flux.MaxPool}
    cv5::Conv
    cv6::Conv
    cv7::Conv
end

# CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
function SPPCSPC(c1, c2; n=1, shortcut=false, groups=1, e=0.5, k=(5, 9, 13))
        c_ = Int(2 * c2 * e)  # hidden channels
        cv1 = Conv(Flux.Conv((1, 1), c1 => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv2 = Conv(Flux.Conv((1, 1), c1 => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv3 = Conv(Flux.Conv((3, 3), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv4 = Conv(Flux.Conv((1, 1), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        m = [Flux.MaxPool((x, x); stride=1, pad=x÷2) for x in k]
        cv5 = Conv(Flux.Conv((1, 1), 4 * c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv6 = Conv(Flux.Conv((3, 3), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv7 = Conv(Flux.Conv((1, 1), 2 * c_ => c2; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c2))

        return SPPCSPC(cv1, cv2, cv3, cv4, m, cv5, cv6, cv7)
end

function SPPCSPC(d::OrderedDict{Any, Any})
    return d["51"]
end

function fuse(m::SPPCSPC)
    cv1 = fuse(m.cv1)
    cv2 = fuse(m.cv2)
    cv3 = fuse(m.cv3)
    cv4 = fuse(m.cv4)
    cv5 = fuse(m.cv5)
    cv6 = fuse(m.cv6)
    cv7 = fuse(m.cv7)

    return SPPCSPC(cv1, cv2, cv3, cv4, m.m, cv5, cv6, cv7)
end

function (m::SPPCSPC)(x::AbstractArray)
    x1 = m.cv4(m.cv3(m.cv1(x)))
    c1 = cat(x1, [mp(x1) for mp in m.m]...; dims=3)
    y1 = m.cv6(m.cv5(c1))
    y2 = m.cv2(x)
    return m.cv7(cat(y1, y2; dims=3))
end

function (m::SPPCSPC)(x::Dict)
    ret = m(x[:x])
    x[:x] = ret
    x[:sppcspc] = x[:x]
    
    return x
end

Flux.@functor SPPCSPC

struct RepConv
    rbr_identity::Union{Nothing, BatchNorm}
    rbr_dense::Chain
    rbr_1x1::Chain
    rbr_reparam::Union{Nothing, Flux.Conv}
    act::Function
end

function RepConv(c1::Int, c2::Int; k=3, s=1, p=nothing, groups=1, act=true)
    # groups = g
    # ch_in = c1
    # ch_out = c2

    activation = act ? silu : identity

    rbr_identity = c1 == c2 && s == 1 ? BatchNorm(c1) : nothing
    rbr_dense = Chain(
        Flux.Conv((k, k), c1 => c2; stride=s, pad=SamePad(), groups=groups, bias=false),
        BatchNorm(c2)
    )
    rbr_1x1 = Chain(
        Flux.Conv((1, 1), c1 => c2; stride=s, pad=0, groups=groups, bias=false),
        BatchNorm(c2)
    )
    rbr_reparam = nothing

    return RepConv(rbr_identity, rbr_dense, rbr_1x1, rbr_reparam, activation)
end

function fuse(m::RepConv)
    rbr_dense = fuse(m.rbr_dense[1], m.rbr_dense[2])
    rbr_1x1 = fuse(m.rbr_1x1[1], m.rbr_1x1[2])
    
    bias_1x1 = rbr_1x1.bias
    weight_1x1_expanded = pad_zeros(rbr_1x1.weight, (1, 1, 1, 1); dims=(1, 2))

    if m.rbr_identity === nothing
        bias_identity_expanded = zeros_like(bias_1x1)
        weight_identity_expanded = zeros_like(weight_1x1_expanded)
    else
        exit(0)
    end

    weight_reparam = rbr_dense.weight .+ weight_1x1_expanded .+ weight_identity_expanded
    bias_reparam = rbr_dense.bias .+ bias_1x1 .+ bias_identity_expanded

    rbr_reparam = Flux.Conv(weight_reparam, bias_reparam; stride=rbr_dense.stride, pad=rbr_dense.pad, dilation=rbr_dense.dilation)

    return RepConv(nothing, m.rbr_dense, m.rbr_1x1, rbr_reparam, m.act)
end

function RepConv(w::Tuple{AbstractArray, AbstractArray}, b::NTuple)
    # groups = g
    # ch_in = c1
    # ch_out = c2
    activation = silu

    rbr_identity = nothing
    β, γ, μ, σ² = b
    rbr_dense = Chain(
        Flux.Conv(w[1], false; stride=1, pad=SamePad()),
        Flux.BatchNorm(identity, β[1], γ[1], μ[1], σ²[1], ϵ, momentum, true, true, true, length(γ[1]))
    )
    rbr_1x1 = Chain(
        Flux.Conv(w[2], false; stride=1, pad=SamePad()),
        Flux.BatchNorm(identity, β[2], γ[2], μ[2], σ²[2], ϵ, momentum, true, true, true, length(γ[2]))
    )
    rbr_reparam = nothing

    return RepConv(rbr_identity, rbr_dense, rbr_1x1, rbr_reparam, activation)
end

function (m::RepConv)(x::AbstractArray)
    id_out = 0
    if m.rbr_identity !== nothing
        id_out = m.rbr_identity(x)
    end

    if m.rbr_reparam !== nothing
        return m.act(m.rbr_reparam(x))
    end

    # println(size(m.rbr_dense(x)))
    # println(size(m.rbr_1x1(x)))
    # println(size(id_out))

    return m.act(m.rbr_dense(x) .+ m.rbr_1x1(x) .+ id_out)
end

Flux.@functor RepConv

struct YOLOv7BackboneBlock
    depth::Int
    mp::MaxPool
    c1::Conv
    c2::Conv
    c3::Conv
    c4::Conv
    c5::Conv
    c6::Conv
    c7::Conv
    c8::Conv
    c9::Conv
    c10::Conv
end

function YOLOv7BackboneBlock(depth::Int64; half_cut=false, start_mp=true)
    if half_cut
        half_depth = depth ÷ 2
    else
        half_depth = depth
    end

    if start_mp
        mp = MaxPool((2, 2))
    else
        mp = MaxPool((1, 1))
    end

    c1 = Conv(2*depth => depth, 1, 1)
    c2 = Conv(2*depth => depth, 1, 1)
    c3 = Conv(depth => depth, 3, 2)
    # cat1
    c4 = Conv(2*depth => half_depth, 1, 1)
    c5 = Conv(2*depth => half_depth, 1, 1)
    c6 = Conv(half_depth => half_depth, 3, 1)
    c7 = Conv(half_depth => half_depth, 3, 1)
    c8 = Conv(half_depth => half_depth, 3, 1)
    c9 = Conv(half_depth => half_depth, 3, 1)
    # cat2
    c10 = Conv(4*half_depth => 4*half_depth, 1, 1)

    return YOLOv7BackboneBlock(depth, mp, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)
end

function YOLOv7BackboneBlock(depth, d::OrderedDict{Any, Any}; off=0)
    mp = d["$(12+off)"]
    cs = [d["$(i+off)"] for i in filter(x -> x != 16 && x != 23, 13:24)]

    return YOLOv7BackboneBlock(depth, mp, cs...)
end

function fuse(m::YOLOv7BackboneBlock)
    c1 = fuse(m.c1)
    c2 = fuse(m.c2)
    c3 = fuse(m.c3)
    c4 = fuse(m.c4)
    c5 = fuse(m.c5)
    c6 = fuse(m.c6)
    c7 = fuse(m.c7)
    c8 = fuse(m.c8)
    c9 = fuse(m.c9)
    c10 = fuse(m.c10)

    return YOLOv7BackboneBlock(m.depth, m.mp, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)
end

function(m::YOLOv7BackboneBlock)(x::AbstractArray)
    xc1 = m.c1(m.mp(x))
    xc3 = m.c3(m.c2(x))
    cat1 = cat(xc3, xc1; dims=3)
    xc4 = m.c4(cat1)
    xc5 = m.c5(cat1)
    xc7 = m.c7(m.c6(xc5))
    xc9 = m.c9(m.c8(xc7))
    cat2 = cat(xc9, xc7, xc5, xc4; dims=3)
    xc10 = m.c10(cat2)
    
    return xc10
end

Flux.@functor YOLOv7BackboneBlock (mp, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,)

struct YOLOv7BackboneInit
    depth::Int
    c1::Conv
    c2::Conv
    c3::Conv
    c4::Conv
    c5::Conv
    c6::Conv
    c7::Conv
    c8::Conv
end

function YOLOv7BackboneInit(depth::Int64)
    c1 = Conv(depth => 2*depth, 3, 2)
    c2 = Conv(2*depth => depth, 1, 1)
    c3 = Conv(2*depth => depth, 1, 1)
    c4 = Conv(depth => depth, 3, 1)
    c5 = Conv(depth => depth, 3, 1)
    c6 = Conv(depth => depth, 3, 1)
    c7 = Conv(depth => depth, 3, 1)
    c8 = Conv(4*depth => 4*depth, 1, 1)

    return YOLOv7BackboneInit(depth, c1, c2, c3, c4, c5, c6, c7, c8)
end

function YOLOv7BackboneInit(depth, d::OrderedDict{Any, Any})
    cs = [d["$i"] for i in filter(x -> x!=10, 3:11)]

    return YOLOv7BackboneInit(depth, cs...)
end

function fuse(m::YOLOv7BackboneInit)
    c1 = fuse(m.c1)
    c2 = fuse(m.c2)
    c3 = fuse(m.c3)
    c4 = fuse(m.c4)
    c5 = fuse(m.c5)
    c6 = fuse(m.c6)
    c7 = fuse(m.c7)
    c8 = fuse(m.c8)

    return YOLOv7BackboneInit(m.depth, c1, c2, c3, c4, c5, c6, c7, c8)
end

function(m::YOLOv7BackboneInit)(x::AbstractArray)
    xc1 = m.c1(x)
    xc2 = m.c2(xc1)
    xc3 = m.c3(xc1)
    xc5 = m.c5(m.c4(xc3))
    xc7 = m.c7(m.c6(xc5))
    cat1 = cat(xc7, xc5, xc3, xc2; dims=3)
    xc8 = m.c8(cat1)
    
    return xc8
end

Flux.@functor YOLOv7BackboneInit (c1, c2, c3, c4, c5, c6, c7, c8,)

struct YOLOv7Backbone
    c1::Conv
    c2::Conv
    c3::Conv
    ybi::YOLOv7BackboneInit
    ybb1::YOLOv7BackboneBlock
    ybb2::YOLOv7BackboneBlock
    ybb3::YOLOv7BackboneBlock
    p3::Bool
    p4::Bool
end

function YOLOv7Backbone(;p3=false, p4=false)
    c1 = Conv(3=>32, 3, 1)
    c2 = Conv(32=>64, 3, 2)
    c3 = Conv(64=>64, 3, 1)

    ybi = YOLOv7BackboneInit(64)
    ybb1 = YOLOv7BackboneBlock(128)
    ybb2 = YOLOv7BackboneBlock(256)
    ybb3 = YOLOv7BackboneBlock(512; half_cut=true)

    return YOLOv7Backbone(c1, c2, c3, ybi, ybb1, ybb2, ybb3, p3, p4)
end

function YOLOv7Backbone(d::OrderedDict{Any, Any}; p3=false, p4=false)
    cs = [d["$i"] for i in 0:2]

    ybi = YOLOv7BackboneInit(64, d)
    ybb1 = YOLOv7BackboneBlock(128, d; off=0)
    ybb2 = YOLOv7BackboneBlock(256, d; off=13)
    ybb3 = YOLOv7BackboneBlock(512, d; off=26)

    return YOLOv7Backbone(cs..., ybi, ybb1, ybb2, ybb3, p3, p4)
end

function fuse(m::YOLOv7Backbone)
    c1 = fuse(m.c1)
    c2 = fuse(m.c2)
    c3 = fuse(m.c3)
    ybi = fuse(m.ybi)
    ybb1 = fuse(m.ybb1)
    ybb2 = fuse(m.ybb2)
    ybb3 = fuse(m.ybb3)

    return YOLOv7Backbone(c1, c2, c3, ybi, ybb1, ybb2, ybb3, m.p3, m.p4)
end

function(m::YOLOv7Backbone)(x::AbstractArray)
    xp3 = m.ybb1(m.ybi(m.c3(m.c2(m.c1(x)))))
    xp4 = m.ybb2(xp3)
    ret = m.ybb3(xp4)

    ret = Dict(:x=>ret)

    if m.p3
        ret[:p3] = xp3
    end
    if m.p4
        ret[:p4] = xp4
    end

    return ret
end

Flux.@functor YOLOv7Backbone (c1, c2, c3, ybi, ybb1, ybb2, ybb3,)

struct YOLOv7HeadRouteback
    depth::Int
    c1::Conv
    up::Upsample
    cback::Conv
    routeback::Symbol
end

function YOLOv7HeadRouteback(depth::Int, routeback::Symbol)
    c1 = Conv(depth => depth ÷ 2, 1, 1)
    up = Upsample(2, :nearest)
    cback = Conv(2*depth => depth ÷ 2, 1, 1)

    return YOLOv7HeadRouteback(depth, c1, up, cback, routeback)
end

function YOLOv7HeadRouteback(depth, routeback, d::OrderedDict{Any, Any}; off=0)
    c1 = d["$(52+off)"]
    up = d["$(53+off)"]
    cback = d["$(54+off)"]

    return YOLOv7HeadRouteback(depth, c1, up, cback, routeback)
end

function fuse(m::YOLOv7HeadRouteback)
    c1 = fuse(m.c1)
    cback = fuse(m.cback)

    return YOLOv7HeadRouteback(m.depth, c1, m.up, cback, m.routeback)
end

function(m::YOLOv7HeadRouteback)(x::Dict)
    xup = m.up(m.c1(x[:x]))
    xrb = m.cback(x[m.routeback])

    x[:x] = cat(xrb, xup; dims=3)
    return x
end

Flux.@functor YOLOv7HeadRouteback (c1, up, cback,)

struct YOLOv7HeadBlock
    depth::Int
    name::Symbol
    c1::Conv
    c2::Conv
    c3::Conv
    c4::Conv
    c5::Conv
    c6::Conv
    c7::Conv
end

function YOLOv7HeadBlock(depth::Int64, name::Symbol) # 256
    half_depth = depth ÷ 2

    c1 = Conv(2*depth => depth, 1, 1)
    c2 = Conv(2*depth => depth, 1, 1)
    c3 = Conv(depth => half_depth, 3, 1)
    c4 = Conv(half_depth => half_depth, 3, 1)
    c5 = Conv(half_depth => half_depth, 3, 1)
    c6 = Conv(half_depth => half_depth, 3, 1)
    # cat1
    c7 = Conv(4*half_depth + 2*depth => depth, 1, 1)

    return YOLOv7HeadBlock(depth, name, c1, c2, c3, c4, c5, c6, c7)
end

function YOLOv7HeadBlock(depth::Int64, name::Symbol, d::OrderedDict{Any, Any}; off=0)
    cs = [d["$(i+off)"] for i in filter(x -> x != 62, 56:63)]
    
    YOLOv7HeadBlock(depth, name, cs...)
end

function fuse(m::YOLOv7HeadBlock)
    c1 = fuse(m.c1)
    c2 = fuse(m.c2)
    c3 = fuse(m.c3)
    c4 = fuse(m.c4)
    c5 = fuse(m.c5)
    c6 = fuse(m.c6)
    c7 = fuse(m.c7)

    return YOLOv7HeadBlock(m.depth, m.name, c1, c2, c3, c4, c5, c6, c7)
end

function(m::YOLOv7HeadBlock)(x::AbstractArray)
    xc1 = m.c1(x)
    xc2 = m.c2(x)
    xc3 = m.c3(xc2)
    xc4 = m.c4(xc3)
    xc5 = m.c5(xc4)
    xc6 = m.c6(xc5)
    cat1 = cat(xc6, xc5, xc4, xc3, xc2, xc1; dims=3)
    xc7 = m.c7(cat1)
    
    return xc7
end

function(m::YOLOv7HeadBlock)(x::Dict)
    ret = m(x[:x])

    x[:x] = ret
    x[m.name] = ret
    
    # Both original x and ret have a :x key
    # For the merge function, elements with the same key, 
    # the value for that key will be of the last Dict listed (ret)
    # return merge(x, ret)
    return x
end

Flux.@functor YOLOv7HeadBlock (c1, c2, c3, c4, c5, c6, c7,)

Flux.flatten(x::Dict) = Flux.flatten(x[:x])

struct YOLOv7HeadIncep
    depth::Int
    mp::MaxPool
    c1::Conv
    c2::Conv
    c3::Conv
    routeback::Symbol
end

function YOLOv7HeadIncep(depth::Int, routeback::Symbol)
    mp = MaxPool((2, 2))
    c1 = Conv(depth => depth, 1, 1)
    c2 = Conv(depth => depth, 1, 1)
    c3 = Conv(depth => depth, 3, 2)

    return YOLOv7HeadIncep(depth, mp, c1, c2, c3, routeback)
end

function YOLOv7HeadIncep(depth::Int, routeback::Symbol, d::OrderedDict{Any, Any}; off=0)
    mp = d["$(76+off)"]
    cs = [d["$(77+i+off)"] for i in 0:2]

    return YOLOv7HeadIncep(depth, mp, cs..., routeback)
end

function fuse(m::YOLOv7HeadIncep)
    c1 = fuse(m.c1)
    c2 = fuse(m.c2)
    c3 = fuse(m.c3)

    return YOLOv7HeadIncep(m.depth, m.mp, c1, c2, c3, m.routeback)
end

function(m::YOLOv7HeadIncep)(x::Dict)
    xc1 = m.c1(m.mp(x[:x]))
    xc2 = m.c2(x[:x])
    xc3 = m.c3(xc2)

    x[:x] = cat(xc3, xc1, x[m.routeback]; dims=3)
    return x
end

Flux.@functor YOLOv7HeadIncep (mp, c1, c2, c3,)

struct YOLOv7HeadTailRepConv
    depth::Int
    repc1::RepConv
    repc2::RepConv
    repc3::RepConv
    routeback1::Symbol
    routeback2::Symbol
    routeback3::Symbol
end

function YOLOv7HeadTailRepConv(depth::Int, routeback1::Symbol, routeback2::Symbol, routeback3::Symbol)
    repc1 = RepConv(depth, 2*depth, k=3, s=1)
    repc2 = RepConv(2*depth, 4*depth, k=3, s=1)
    repc3 = RepConv(4*depth, 8*depth, k=3, s=1)

    return YOLOv7HeadTailRepConv(depth, repc1, repc2, repc3, routeback1, routeback2, routeback3)
end

function YOLOv7HeadTailRepConv(depth::Int, routeback1::Symbol, routeback2::Symbol, routeback3::Symbol, d::OrderedDict{Any, Any})
    return YOLOv7HeadTailRepConv(depth, d["102"], d["103"], d["104"], routeback1, routeback2, routeback3)
end

function fuse(m::YOLOv7HeadTailRepConv)
    repc1 = fuse(m.repc1)
    repc2 = fuse(m.repc2)
    repc3 = fuse(m.repc3)

    return YOLOv7HeadTailRepConv(m.depth, repc1, repc2, repc3, m.routeback1, m.routeback2, m.routeback3)
end

function(m::YOLOv7HeadTailRepConv)(x::Dict)
    xrepc1 = m.repc1(x[m.routeback1])
    xrepc2 = m.repc2(x[m.routeback2])
    xrepc3 = m.repc3(x[m.routeback3])

    return [xrepc1, xrepc2, xrepc3]
end

Flux.@functor YOLOv7HeadTailRepConv (repc1, repc2, repc3,)

struct ImplicitAddition
    w
end

function ImplicitAddition(depth::Int; mean=0.0, std=0.02, device=gpu)
    d = Normal(mean, std)
    w = Float32.(rand(d, 1, 1, depth, 1)) |> device

    return ImplicitAddition(w)
end

function (m::ImplicitAddition)(x::AbstractArray)
    return m.w .+ x
end

Flux.@functor ImplicitAddition (w,)

struct ImplicitMultiplication
    w
end

function ImplicitMultiplication(depth::Int; mean=0.0, std=0.02, device=gpu)
    d = Normal(mean, std)
    w = Float32.(rand(d, 1, 1, depth, 1)) |> device

    return ImplicitMultiplication(w)
end

function (m::ImplicitMultiplication)(x::AbstractArray)
    return x .* m.w
end

Flux.@functor ImplicitMultiplication (w,)

struct IDetec
    classes::Int
    outputs::Int
    detec_layers::Int
    n_anchors::Int
    out_conv
    ia
    im
    anchors
    anchor_grid
end

function fuse(m::IDetec)
    out_conv = []
    ia = []
    im = []
    for i in 1:length(m.out_conv)
        # Fuse ImplicitA and Conv
        bias = m.out_conv[i].bias .+ (m.ia[i].w[1, :, :, 1] * m.out_conv[i].weight[1, 1, :, :])'
        ia_w = zeros_like(m.ia[i].w)

        # Fuse ImplicitM and Conv
        bias .*= m.im[i].w[1, 1, :, 1]
        weight = m.out_conv[i].weight .* reshape(m.im[i].w, (1, 1, 1, length(m.im[i].w)))
        im_w = ones_like(m.im[i].w)

        push!(out_conv, Flux.Conv(weight, bias[:, 1]; stride=m.out_conv[i].stride, pad=m.out_conv[i].pad, dilation=m.out_conv[i].dilation))
        push!(ia, ImplicitAddition(ia_w))
        push!(im, ImplicitMultiplication(im_w))
    end

    return IDetec(m.classes, m.outputs, m.detec_layers, m.n_anchors,
                    Tuple(out_conv), Tuple(ia), Tuple(im), m.anchors, m.anchor_grid)
end

function IDetec(classes::Int; anchors=(), anchor_grid=(), channels::Tuple{Vararg{Int}}=())
    outputs = classes + 5
    detec_layers = length(anchors)
    n_anchors = length(anchors[1]) ÷ 2

    init_grid = [Flux.zeros32(1) for _ in 1:detec_layers]
    out_conv = Tuple([Flux.Conv((1, 1), ch => outputs * n_anchors) for ch in channels])

    ia = Tuple([ImplicitAddition(ch) for ch in channels])
    im = Tuple([ImplicitMultiplication(outputs * n_anchors) for _ in channels])

    return IDetec(classes, outputs, detec_layers, n_anchors, out_conv, ia, im, anchors, anchor_grid)
end

function IDetec(d::OrderedDict{Any, Any})
    d["105"]
end

function (m::IDetec)(x::Vector{CuArray{Float32, 4, CUDA.Mem.DeviceBuffer}})
    # z = []
    # println(typeof(x))
    # println(size(x))
    # y = [Flux.zeros32(1, 1, 1, 1, 1), Flux.zeros32(1, 1, 1, 1, 1), Flux.zeros32(1, 1, 1, 1, 1)] |> gpu
    # println(size(y))
    y = [
        let
            # println(size(x[i]), " ", typeof(x[i]))
            # println(size(m.ia[i].w), " ", typeof(m.ia[i].w))
            # x[i] = m.out_conv[i](m.ia[i](x[i]))
            r = m.ia[i](x[i])

            # println(size(r), " ", typeof(r))
            # println(size(m.out_conv[i].weight))
            # println(size(m.out_conv[i].bias))
            r = m.out_conv[i](r)

            # println(size(r), typeof(r))
            
            # x[i] = m.im[i](x[i])
            r = m.im[i](r)
            # println(size(r), typeof(r))

            nx, ny, _, bs = size(r)
            # println((nx, ny, m.outputs, m.n_anchors, bs))
            # println(size(r))
            permutedims(reshape(r, (nx, ny, m.outputs, m.n_anchors, bs)), (3, 1, 2, 4, 5))
            # permutedims(reshape(r, (bs, m.n_anchors, m.outputs, ny, nx)), (3, 4, 5, 2, 1))
        end
        for i in 1:m.detec_layers
    ]

    return y
end

Flux.@functor IDetec (out_conv, ia, im,)

# function Flux.trainable(a::IDetec)
#     ias = (ia1=a.ia[1], ia2=a.ia[2], ia3=a.ia[3])
#     ims = (im1=a.im[1], im2=a.im[2], im3=a.im[3])
#     convs = (out_conv1=a.out_conv[1], out_conv2=a.out_conv[2], out_conv3=a.out_conv[3])
#     merge(convs, ias, ims)
# end
