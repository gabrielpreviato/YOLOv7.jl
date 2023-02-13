# (:input, :output, :name, :op_type, :domain, :attribute, :doc_string)
using ProtoBuf

onnx_path="yolov7_training.onnx"
io = open(onnx_path)
d = ProtoDecoder(io)
mp = decode(d, YOLOv7.onnx.ModelProto)
g_raw = mp.graph
g_float = YOLOv7.onnx.get_array.(g_raw.initializer)
g_name = [x.name for x in g_raw.initializer]
g = Dict([(k,v) for (k,v) in zip(g_name, g_float)])
# g = ONNX.convert(f.graph)