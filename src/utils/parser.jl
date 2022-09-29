using YAML
using Flux

using YOLOv7: YOLOChain

struct YOLOBlock{Block}
    b::Block
end

function parse_yolo(file)
    yolo_yaml = YAML.load_file(file; dicttype=Dict{Symbol,Any})

    chain = YOLOChain([])
    for node in [yolo_yaml[:backbone] ; yolo_yaml[:head]]
        m = load_node(chain, YOLOBlock(Symbol(node[3])), node[1], node[2], node[4])
        push!(chain.layers, m)
    end

    print(chain)
end

function load_node(chain::YOLOChain, ::YOLOBlock{:Conv}, from::Int, number::Int, args::Vector{Int})
    @assert length(args) == 3
    
    ch_in = chain[from]

    ch_out = args[1]
    filter = args[2]
    stride = args[3]
    
    return Flux.Conv(filter, ch_in => ch_out; stride=stride) 
end

function load_node(chain::YOLOChain, ::YOLOBlock{:Concat}, from::Vector{Int}, number::Int, args::Vector{Int})
    @assert length(args) == 1
    @assert number == 1
    
    dim = 4 - args[1]
    return Flux.Join(x -> cat(x; dims=dim), chain[from]...)
end
