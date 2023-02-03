using Flux
using NNlib

function get_filter(filt_size::Int)
    if filt_size == 1
        a = [1., ]
    elseif filt_size == 2
        a = [1., 1.]
    elseif filt_size == 3
        a = [1., 2., 1.]
    elseif filt_size == 4
        a = [1., 3., 3., 1.]
    elseif filt_size == 5
        a = [1., 4., 6., 4., 1.]
    elseif filt_size == 6
        a = [1., 5., 10., 10., 5., 1.]
    elseif filt_size == 7
        a = [1., 6., 15., 20., 15., 6., 1.]
    end

    filt = a * a'
    filt = filt / sum(filt)

    return Float32.(filt)
end

struct PadReflect 
    pad::Function
end

PadReflect(size::Int) = PadReflect(x -> NNlib.pad_reflect(x, size))

(m::PadReflect)(x::AbstractArray) = m.pad(x)

struct DownSample
    stride::Int
    pad_off::Int
    pad::Function
    filt::AbstractArray
    filt_size::Int
end

function DownSample(channels::Int; pad_class=PadReflect, pad_off=0, stride=2, filt_size=3)
    filt = get_filter(filt_size)
    filt = repeat(filt, 1, 1, 1, channels)

    pad_size = Int(ceil((filt_size - 1) / 2)) + pad_off
    pad = x -> pad_class(pad_size)(x)

    return DownSample(stride, pad_off, pad, filt, filt_size)
end

function (m::DownSample)(x::AbstractArray)
    if m.filt_size == 1
        if m.pad_off == 0
            return x[1:m.stride:end, 1:m.stride:end, :, :]
        else
            return m.pad(x)[1:m.stride:end, 1:m.stride:end, :, :]
        end
    else
        conv(m.pad(x), m.filt; stride=m.stride, groups=size(x)[3])
    end
end

function resnet_generator(input_nc::Int, ngf::Int; n_downsampling::Int=2, antialias::Bool=true)
    model = [
        PadReflect(3),
        Flux.Conv((7, 7), input_nc => ngf; pad=0),
        Flux.InstanceNorm(ngf),
        Flux.relu
    ]

    for i in 1:n_downsampling
        mult = 2 ^ (i-1)

        if antialias
            m = [
                Flux.Conv((3, 3), ngf*mult => ngf*mult*2; stride=2, pad=SamePad()),
                Flux.InstanceNorm(ngf*mult*2),
                Flux.relu,
                DownSample(ngf*mult*2)
            ]
        else
            m = []
        end
    end
end