struct YOLOBlock{Block} end

struct Node
    children::Vector
    parents::Union{Node,Vector}
    op::Any
    ch_out::Union{Int}
    node::Any
    visited::Int
end

Node() = Node([], [], nothing, 0, nothing, 0)

function Node(children, parents, op)
    args = op[4]
    println(op[1])
    if op[1] == YOLOBlock{:Upsample}()
        println("in if")
        if typeof(parents) === YOLOv7.Node
            parents = [parents]
        end
        ch_out = sum(x -> x.ch_out, parents; init=0)
    elseif op[1] == YOLOBlock{:SPPCSPC}()
        ch_out = args[1]
    else
        if length(args) == 3
            ch_out = args[1]
        else
            if typeof(parents) === YOLOv7.Node
                parents = [parents]
            end

            if length(parents) > 0
                println(parents)
                ch_out = sum(x -> x.ch_out, parents)
            else
                ch_out = 0
            end
        end
    end

    if length(children) == 1
        children = [children]
    end

    println(typeof(children), children)

    return Node(children, parents, op, ch_out, nothing, 0)
end

function Base.show(io::IO, m::Node)
    print(io, m.op, length(m.children))
end