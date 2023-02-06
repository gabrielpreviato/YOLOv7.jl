using YOLOv7
using Graphs, GraphPlot

m = YOLOv7.load_model(raw"src\conf\yolov7.yaml")

g = Graphs.SimpleGraphs.SimpleDiGraph()

add_vertices!(g, length(m.nodes))
for (i, node) in enumerate(m.nodes)
    println(node.op)
    for f in node.op[2]
        add_edge!(g, f-1, i)
    end
end

nodelabel = 1:nv(g)
nodesize = [2.0 for _ in 1:nv(g)]
nodesize[1] = nodesize[end] = 3.5
layout=(args...)->spring_layout(args...; C=2.0)
gplot(g, nodelabel=nodelabel, layout=layout, nodesize=nodesize)

