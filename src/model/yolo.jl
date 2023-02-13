using Flux
using ProtoBuf

# abstract type _yolov7 end

# abstract type _yolov7_small end

# abstract type _yolov7_big end

struct yolov7
    name::String
    pretrained::Bool
    m
end

function yolov7(;name="yolov7", onnx_path="yolov7_training.onnx", pretrained=false)
    anchors = Tuple([
    [[0.87646 0.76221]; [0.30786 3.21289]; [5.64453 0.39526]],
    [[0.27368 2.20898]; [0.39478 3.63867]; [4.15234 0.60596]],
    [[0.32495 2.86133]; [0.59619 4.11719]; [2.95312 1.13184]]
])

    g = nothing
    if pretrained
        io = open(onnx_path)
        d = ProtoDecoder(io)
        mp = decode(d, YOLOv7.onnx.ModelProto)
        g_raw = mp.graph
        g_float = YOLOv7.onnx.get_array.(g_raw.initializer)
        g_name = [x.name for x in g_raw.initializer]
        g = Dict([(k,v) for (k,v) in zip(g_name, g_float)])
        # g = ONNX.convert(f.graph)
    end

    model = Chain(                                                                                                                                                                                  
        YOLOv7.YOLOv7Backbone(g, pretrained; p3=true, p4=true),
        YOLOv7.SPPCSPC(1024, 512, g, pretrained),
        YOLOv7.YOLOv7HeadRouteback(512, :p4, g, pretrained; off=0),
        YOLOv7.YOLOv7HeadBlock(256, :h1, g, pretrained; off=0), # 63
        YOLOv7.YOLOv7HeadRouteback(256, :p3, g, pretrained; off=12),
        YOLOv7.YOLOv7HeadBlock(128, :h2, g, pretrained; off=12), # 75
        YOLOv7.YOLOv7HeadIncep(128, :h1, g, pretrained; off=0),
        YOLOv7.YOLOv7HeadBlock(256, :h3, g, pretrained; off=25), #88
        YOLOv7.YOLOv7HeadIncep(256, :sppcspc, g, pretrained; off=13),
        YOLOv7.YOLOv7HeadBlock(512, :h4, g, pretrained; off=38), #101
        YOLOv7.YOLOv7HeadTailRepConv(128, :h2, :h3, :h4, g, pretrained),
        YOLOv7.IDetec(80, g, pretrained; anchors=anchors, channels=(256, 512, 1024)),
    )

    # if pretrained
    #     load_weights(model)
    # end

    return yolov7(name, pretrained, model)
end

function load_weights(model::Chain, )
    f = ONNX.readproto(open(onnx_path), ONNX.Proto.ModelProto())
    g = ONNX.convert(f.graph)

    for block in model
        load_weights(block, g)
    end
end

function load_weights(model, g)
    return
end

function load_weights(model::YOLOv7Backbone, g)
    model.c1.conv.weight = g.initializer["model.model.0.conv.weight"]
    model.c2.conv.weight = g.initializer["model.model.1.conv.weight"]
    model.c3.conv.weight = g.initializer["model.model.2.conv.weight"]

    model.c1.conv.bias = g.initializer["model.model.0.conv.bias"]
    model.c2.conv.bias = g.initializer["model.model.1.conv.bias"]
    model.c3.conv.bias = g.initializer["model.model.2.conv.bias"]

    load_weights(model.ybi, g)
    
    load_weights(model.ybb1, 0, g)
    load_weights(model.ybb2, 13, g)
    load_weights(model.ybb3, 26, g)
end

function load_weights(model::YOLOv7BackboneInit, g)
    model.c1.conv.weight = g.initializer["model.model.3.conv.weight"]
    model.c2.conv.weight = g.initializer["model.model.4.conv.weight"]
    model.c3.conv.weight = g.initializer["model.model.5.conv.weight"]
    model.c4.conv.weight = g.initializer["model.model.6.conv.weight"]
    model.c5.conv.weight = g.initializer["model.model.7.conv.weight"]
    model.c6.conv.weight = g.initializer["model.model.8.conv.weight"]
    model.c7.conv.weight = g.initializer["model.model.9.conv.weight"]
    model.c8.conv.weight = g.initializer["model.model.11.conv.weight"]
end

function load_weights(model::YOLOv7BackboneBlock, offset, g)
    model.c1.conv.weight = g.initializer["model.model.$(13 + offset).conv.weight"]
    model.c2.conv.weight = g.initializer["model.model.$(14 + offset).conv.weight"]
    model.c3.conv.weight = g.initializer["model.model.$(15 + offset).conv.weight"]
    model.c4.conv.weight = g.initializer["model.model.$(17 + offset).conv.weight"]
    model.c5.conv.weight = g.initializer["model.model.$(18 + offset).conv.weight"]
    model.c6.conv.weight = g.initializer["model.model.$(19 + offset).conv.weight"]
    model.c7.conv.weight = g.initializer["model.model.$(20 + offset).conv.weight"]
    model.c8.conv.weight = g.initializer["model.model.$(21 + offset).conv.weight"]
    model.c9.conv.weight = g.initializer["model.model.$(22 + offset).conv.weight"]
    model.c10.conv.weight = g.initializer["model.model.$(24 + offset).conv.weight"]
end