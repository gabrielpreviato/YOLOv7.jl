using YOLOv7

using Flux
using Images
using MLUtils
using Test

@testset "yolov7 constructors" begin
    # msg = """The GPU function is being called but the GPU is not accessible. 
    #         Defaulting back to the CPU. (No action is required if you want to run on the CPU)."""
    # @test_logs (:info, msg) yolov7()
    @test_nowarn yolov7(name="yolo test", nc=2, class_names=["car", "ball"])
    
    # Usage of non-default anchors
    anchor_grid = [
        [[12 16]; [19 36]],
        [[36 75]; [76 55]],
        [[142 110]; [192 243]]
    ]
    @test_nowarn yolov7(name="yolo test", nc=2, class_names=["car", "ball"], anchor_grid=anchor_grid)
    
    anchors = [
        [[12 16]],
        [[36 75]],
        [[142 110]]
    ] ./ [8, 16, 32]
    @test_nowarn yolov7(name="yolo test", nc=2, class_names=["car", "ball"], anchors=anchors)
    
    @test_throws AssertionError yolov7(nc=2, class_names=["obstacle"])
end

@testset "yolov7_from_torch constructors" begin
    @test_nowarn yolov7_from_torch()
    @test_nowarn yolov7_from_torch(nc=2, class_names=["car", "ball"])
    
    @test_throws SystemError yolov7_from_torch(name="yolo test", pickle_path="")
end



@testset "Pretrained non-fused yolov7" begin
    horse_answer = [
        [0.178711   359.38      0.0657387   195.533     305.919]
        [238.836    264.74      233.804     244.331     261.715]
        [261.353    493.152     127.619     346.86      378.435]
        [513.818    434.934     357.73      454.718     403.914]
        [0.951704   0.937895    0.887299    0.790568    0.681191]
        [18.0       18.0        18.0        18.0        18.0]
    ]

    model = yolov7_from_torch()
    m = model.m

    x = float.(Images.load.(["$(@__DIR__)/images/horses.jpg"]))
    x = imresize.(x, 640, 640)
    x = channelview.(x)
    x = permutedims.(x, ((3, 2, 1),))
    x = stack(x, dims=4)

    ŷ = m(x)

    out = output_to_box(ŷ, model.anchor_grid, model.strid)
    out_nms = non_max_suppression([out]; nc=80)[1]
    @test out_nms ≈ horse_answer
end

@testset "Pretrained fused yolov7" begin
    horse_answer = [
        [0.178711   359.38      0.0657387   195.533     305.919]
        [238.836    264.74      233.804     244.331     261.715]
        [261.353    493.152     127.619     346.86      378.435]
        [513.818    434.934     357.73      454.718     403.914]
        [0.951704   0.937895    0.887299    0.790568    0.681191]
        [18.0       18.0        18.0        18.0        18.0]
    ]
    fused_model = fuse(yolov7_from_torch())
    fm = fused_model.m

    x = float.(Images.load.(["$(@__DIR__)/images/horses.jpg"]))
    x = imresize.(x, 640, 640)
    x = channelview.(x)
    x = permutedims.(x, ((3, 2, 1),))
    x = stack(x, dims=4)

    fŷ = fm(x)

    out = output_to_box(fŷ, fused_model.anchor_grid, fused_model.strid)
    out_nms = non_max_suppression([out]; nc=80)[1]
    @test out_nms ≈ horse_answer
end
