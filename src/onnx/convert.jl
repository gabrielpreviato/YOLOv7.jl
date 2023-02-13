using YOLOv7.onnx

function get_array(x::TensorProto)
    if (x.data_type == 1)
        if !isempty(x.float_data)
            x = reshape(reinterpret(Float32, x.float_data), reverse(x.dims)...)
        else
            x = reshape(reinterpret(Float32, x.raw_data), reverse(x.dims)...)
        end
        return x
    end
    if x.data_type == 7
        if !isempty(x.raw_data)
            x = reshape(reinterpret(Int64, x.raw_data), reverse(x.dims)...)
        else
            x = reshape(reinterpret(Int64, x.int64_data), reverse(x.dims)...)
        end
        return x
    end
    if x.data_type == 9
        x = reshape(reinterpret(Int8, x.raw_data), reverse(x.dims)...)
        return x
    end
    if x.data_type == 6
         x = reshape(reinterpret(Int32, x.raw_data), reverse(x.dims)...)
        return x
    end
    if x.data_type == 11
        if !isempty(x.raw_data)
            x = reshape(reinterpret(Float64, x.raw_data), reverse(x.dims)...)
        else
            x = Base.convert(Array{Float32, N} where N, reshape(x.double_data , reverse(x.dims)...))
        end
        return x
    end
    if x.data_type == 10
        x = reshape(reinterpret(Float16, x.raw_data), reverse(x.dims)...)
        return x
    end
end