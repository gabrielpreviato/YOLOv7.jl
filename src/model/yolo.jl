using Flux
using ProtoBuf
using Pickle
using OrderedCollections

# abstract type _yolov7 end

# abstract type _yolov7_small end

# abstract type _yolov7_big end

struct yolov7
    name::String
    pretrained::Bool
    class_names
    m
end

function fuse(m::yolov7)
    layers = []
    for layer in m.m
        push!(layers, fuse(layer))
    end

    return yolov7(m.name, m.pretrained, m.class_names, Chain(layers))
end

function yolov7(;name="yolov7", onnx_path="yolov7_training.onnx", pretrained=false)
    anchors = Tuple([
        [[0.87646 0.76221]; [0.30786 3.21289]; [5.64453 0.39526]],
        [[0.27368 2.20898]; [0.39478 3.63867]; [4.15234 0.60596]],
        [[0.32495 2.86133]; [0.59619 4.11719]; [2.95312 1.13184]]
    ])

    anchor_grid = Tuple([
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
        YOLOv7.IDetec(80, g, pretrained; anchors=anchors, channels=(256, 512, 1024), anchor_grid=anchor_grid),
    )

    # if pretrained
    #     load_weights(model)
    # end

    return yolov7(name, pretrained, [], model)
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

function yolov7_from_torch(;name="yolov7", pickle_path="yolov7_training.pt", nc=80, anchors=(), anchor_grid=(), channels=())
    d::OrderedDict, class_names, nc_model = load_pickle(pickle_path)

    keep_head = false
    if nc_model == nc
        keep_head = true
    end

    model = Chain(                                                                                                                                                                                  
        YOLOv7.YOLOv7Backbone(d; p3=true, p4=true),
        YOLOv7.SPPCSPC(d),
        YOLOv7.YOLOv7HeadRouteback(512, :p4, d; off=0),
        YOLOv7.YOLOv7HeadBlock(256, :h1, d; off=0), # 63
        YOLOv7.YOLOv7HeadRouteback(256, :p3, d; off=12),
        YOLOv7.YOLOv7HeadBlock(128, :h2, d; off=12), # 75
        YOLOv7.YOLOv7HeadIncep(128, :h1, d; off=0),
        YOLOv7.YOLOv7HeadBlock(256, :h3, d; off=25), #88
        YOLOv7.YOLOv7HeadIncep(256, :sppcspc, d; off=13),
        YOLOv7.YOLOv7HeadBlock(512, :h4, d; off=38), #101
        keep_head ? YOLOv7.YOLOv7HeadTailRepConv(128, :h2, :h3, :h4, d) : YOLOv7.YOLOv7HeadTailRepConv(128, :h2, :h3, :h4),
        keep_head ? YOLOv7.IDetec(d) : YOLOv7.IDetec(nc; anchors=anchors, anchor_grid=anchor_grid, channels=channels),
    )

    return yolov7(name, true, class_names, model)
end
