using YAML
using Flux

using YOLOv7: YOLOChain

struct YOLOBlock{Block} end

function parse_yolo(file)
    yolo_yaml = YAML.load_file(file; dicttype=Dict{Symbol,Any})

    chain = YOLOChain([3], Vector{Any}([[]]))
    for node in yolo_yaml[:backbone]
        from = node[1]
        if isa(from, Vector)
            from[from .< 0] .= length(chain) .+ from .+ 1
        else
            if from < 0
                from = length(chain) + from + 1
            end
        end
        m = load_node!(chain, YOLOBlock{Symbol(node[3])}(), from, node[2], node[4])
        println(m)
        # push!(chain.layers, m)
    end

    print(chain)
end

function load_node!(chain::YOLOChain, ::YOLOBlock{:Conv}, from::Int, number::Int, args::Vector{Int})
    @assert length(args) == 3
    
    # Julia does not allow negative index
    ch_in = chain[from]

    ch_out = args[1]
    filter = args[2]
    stride = args[3]

    if length(filter) == 1
        filter = (filter, filter)
    end

    push!(chain.channels, ch_out)

    m = Flux.Conv(filter, ch_in => ch_out; stride=stride) 

    push!(chain.m, m)

    return m
end

function load_node!(chain::YOLOChain, ::YOLOBlock{:Concat}, from::Vector{Int}, number::Int, args::Vector{Int})
    @assert length(args) == 1
    @assert number == 1

    dim = 4 - args[1]

    return Flux.Parallel(x -> cat(x; dims=dim), chain.m[from]...)
end

function load_node!(chain::YOLOChain, ::YOLOBlock{:MP}, from::Int, number::Int, args::Vector{Any})
    if length(args) == 0
        window = (2, 2)
    end

    ch_in = chain[from]
    ch_out = ch_in
    
    push!(chain.channels, ch_out)

    m = Flux.MaxPool(window)
    push!(chain.m, m)

    return m
end