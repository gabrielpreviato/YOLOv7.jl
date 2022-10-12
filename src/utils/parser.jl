using YAML
using Flux

using YOLOv7: Node, Split, YOLOChain, SPPCSPC, silu, YOLOBlock

using YOLOv7

function parse_yolo(file)
    yolo_yaml = YAML.load_file(file; dicttype=Dict{Symbol,Any})

    # chain = YOLOChain([3], Vector{Any}([[]]))
    root = Node([], [], (YOLOBlock{:input}(), [0], 1, []), 3, nothing, 0)
    node_list = [root]
    for node in [yolo_yaml[:backbone]; yolo_yaml[:head]]
        
        from = node[1]
        if isa(from, Vector)
            from[from .< 0] .= length(node_list) .+ from[from .< 0] .+ 1
        else
            if from < 0
                from = length(node_list) + from + 1
            end
            from = [from]
        end

        number = node[2]
        op = node[3]
        args = node[4]

        println(node)
        n = Node([], node_list[from], (YOLOBlock{Symbol(op)}(), from, number, args))
        push!(node_list, n)

        for f in from
            push!(node_list[f].children, n)
        end
        
        # m = load_node!(chain, YOLOBlock{Symbol(node[3])}(), from, node[2], node[4])
        # push!(chain.layers, m)
    end
    
    return node_list
end

function parse_node(graph, root::Node)
    visit_heap::Array{Node} = [root.children[1]]
    layers::Array{Any} = [([], "chain")]

    while !isempty(visit_heap)
        graph_node = popfirst!(visit_heap)
        graph_node.visited += 1
        if graph_node.visited != length(graph_node.parents)
            # continue
        end


        println(graph_node)


        if length(graph_node.children) > 1
            popped_out = popfirst!(layers)
            working_layers, type_layer = popped_out           
            
            flux_node = load_node(graph, graph_node)
            push!(working_layers, flux_node)
            
            if graph_node.op[1] !== YOLOBlock{:Concat}()
                pushfirst!(visit_heap, graph_node.children...)
                pushfirst!(layers, (working_layers, type_layer))
            else
                pushfirst!(visit_heap, graph_node.children...)
                # pushfirst!(layers, (working_layers, type_layer))
            end

            new_parallel = ([], "parallel")
            push!(working_layers, new_parallel)
            pushfirst!(layers, new_parallel)

            # if any(map(x -> x.op[1] == YOLOBlock{:Concat}(), graph_node.children))
            #     push!(new_parallel[1], identity)
            # end
            for c in reverse(graph_node.children)
                if c.op[1] == YOLOBlock{:Concat}()
                    child_chain = (Array{Any}([identity]), "function")
                    println(typeof(child_chain))
                    # pushfirst!(layers, child_chain)
                    push!(new_parallel[1], child_chain)
                else
                    child_chain = ([], "chain")
                    push!(new_parallel[1], child_chain)
                    pushfirst!(layers, child_chain)
                end
            end
            # pushfirst!(layers, ([], "chain"))
        elseif length(graph_node.children) <= 1
            
            if graph_node.op[1] !== YOLOBlock{:Concat}()
                popped_out = popfirst!(layers)
                working_layers, type_layer = popped_out           
                
                if type_layer == "parallel"

                end
                
                flux_node = load_node(graph, graph_node)
                push!(working_layers, flux_node)
                
                # println(flux_node)
                # println(working_layers)
                
                pushfirst!(visit_heap, graph_node.children...)
                pushfirst!(layers, popped_out)
            else
                if length(graph_node.parents) == graph_node.visited
                    popped_out = popfirst!(layers)
                    working_layers, type_layer = popped_out           
                    
                    pushfirst!(visit_heap, graph_node.children...)

                    
                    if length(layers) == 3
                        parallel_layer, type_layer = popfirst!(layers)
                        parallel_layer, type_layer = popfirst!(layers)
                    end
                    if length(layers) == 2
                        parallel_layer, type_layer = popfirst!(layers)
                        # parallel_layer, type_layer = popfirst!(layers)
                    end
                    println("Finished concat")
                    
                else
                    popped_out = popfirst!(layers)
                    working_layers, type_layer = popped_out
                    println(type_layer)    
                    println(working_layers)
                    if type_layer == "chain" && working_layers == []
                        pushfirst!(layers, popped_out)
                    end
                end

                # pushfirst!(layers, (working_layers, type_layer))
                # previous_layers, previous_type_layer = popfirst!(layers)
                # push!(previous_layers, (working_layers, type_layer))
                # pushfirst!(layers, (previous_layers, previous_type_layer))
            end

            if length(graph_node.children) == 1
                if graph_node.children[1].op[1] == YOLOBlock{:Concat}() 
                    println("here")
                    popfirst!(layers)
                end
            end
        end
    end

    return layers
end

function create_chain(layer_tuple::Tuple{Vector{Any}, String})
    layers, chain_type = layer_tuple
    if chain_type == "chain"
        return Flux.Chain(create_chain.(layers))
    elseif chain_type == "parallel"
        return Flux.Parallel((x...) -> cat(x...; dims=3), create_chain.(layers)...)
    elseif chain_type == "function"
        return layers[1]
    end
end

function create_chain(layer_tuple::Tuple{Function, String})
    layers, chain_type = layer_tuple
    println(layers)
    println("/////")
    return layers
    # if chain_type == "identity"
    #     return Flux.Chain(create_chain.(layers))
    # elseif chain_type == "parallel"
    #     return Flux.Parallel(create_chain.(layers)...)
    # end
end

function create_chain(layer_tuple::Nothing)
    return
end

function create_chain(layer_tuple::Array)
    println("call")
    return create_chain.(layer_tuple)
end

function create_chain(layer_tuple::Union{Conv, MaxPool, Function})
    return layer_tuple
end

function parse_graph(graph::Vector{Node})
    layers = []  
    
    s::Node = graph[1]

    layers = parse_node(graph, s)

    println("-------")
    println(length(layers))

    model = create_chain(layers[end])
    return model, layers

    # n = load_node(graph, s.op...)
    # explore = [n]
    # actual_chain = []
    # while !isempty(explore)
    #     u = pop!(explore)
    #     println(u)
    #     push!(actual_chain, u)

    #     if length(u.children) > 1
    #         children_chains, next = parse_children(graph, u)
    #         u = Split(children_chains)
    #         push!(actual_chain, u)
            
    #         push!(layers, actual_chain)
    #         actual_chain = []
    #     else
    #         next = u.children[1]
    #     end

    #     push!(explore, next)
    # end

    # push!(layers, actual_chain)
    # println(layers)
end

load_node(chain, node::Node) = load_node(chain, node.op...)

function load_node(chain, ::YOLOBlock{:input}, from::Vector{Int}, number::Int, args::Vector)
    # return chain[1]
    return nothing
end

function load_node(chain, ::YOLOBlock{:Conv}, from::Vector{Int}, number::Int, args::Vector{Int})
    if length(args) <= 3
        activation = silu
    elseif length(args) == 4
        activation = args[4]
    end
    # Julia does not allow negative index
    ch_in = chain[from][1].ch_out

    ch_out = args[1]
    filter = args[2]
    stride = args[3]

    if length(filter) == 1
        filter = (filter, filter)
    end

    # push!(chain.channels, ch_out)

    conv = Flux.Conv(filter, ch_in => ch_out; stride=stride, pad=Flux.SamePad())
    bn = Flux.BatchNorm(ch_out)
    act = activation

    m = YOLOv7.Conv(conv, bn, act)

    # push!(chain.m, m)

    return m
end

function load_node(chain, ::YOLOBlock{:Concat}, from::Vector{Int}, number::Int, args::Vector{Int})
    @assert length(args) == 1
    @assert number == 1

    dim = 4 - args[1]

    # m = Flux.Parallel((x...) -> cat(x...; dims=dim), map(x -> x.node, chain[from])...)
    # println(m)

    return (x...) -> cat(x...; dims=dim)
end

function load_node(chain, ::YOLOBlock{:MP}, from::Vector{Int}, number::Int, args::Vector{Any})
    if length(args) == 0
        window = (2, 2)
    end

    ch_in = chain[from]
    ch_out = ch_in
    
    # push!(chain.channels, ch_out)

    m = Flux.MaxPool(window)
    # push!(chain.m, m)

    return m
end

function load_node(chain, ::YOLOBlock{:SPPCSPC}, from::Vector{Int}, number::Int, args::Vector{Int64})
    ch_in = chain[from][1].ch_out
    ch_out = args[1]
    
    m = SPPCSPC(ch_in, ch_out)

    return m
end

function load_node(chain, ::YOLOBlock{:Flatten}, from::Vector{Int}, number::Int, args::Vector{Any})
    # ch_in = chain[from][1].ch_out
    # ch_out = args[1]
    
    m = Flux.flatten

    return m
end

function load_node(chain, ::YOLOBlock{:Upsample}, from::Vector{Int}, number::Int, args::Vector{Any})
    scale = args[2]
    mode = Symbol(args[3])
    
    m = Flux.Upsample(scale, mode)

    return m
end

function load_node(chain, ::YOLOBlock{:Dense}, from::Vector{Int}, number::Int, args::Union{Vector{Any}, Vector{Int}})
    ch_in = args[1]
    ch_out = args[2]

    if length(args) >= 3
        activation = args[3]
        if activation == "sigmoid"
            activation = Flux.sigmoid
        elseif activation == "relu"
            activation = Flux.relu
        end
    else
        activation = identity
    end
    
    m = Flux.Dense(ch_in => ch_out, activation)

    return m
end

function load_model()
    l = parse_yolo(raw"C:\Users\Previato\YOLOv7.jl\src\conf\yolov7.yaml")
    b = []
    for i in l
        push!(b, load_node(l, i.op...))
    end

    m = YOLOChain(l, b)
end

# X = rand(Float32, 640, 640, 3, 100)
# Y = rand(0:9, 100)
# data = Flux.DataLoader((X, Y), batchsize=8) 
# loss(x, y) = Flux.Losses.mse(m(x), y)
# ps = Flux.params(m)
# opt = Flux.Adam(0.001, (0.9, 0.8))