using YOLOv7

using Flux
using Test

@testset "yolov7" begin
    @test_nowarn yolov7()
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

@testset "yolov7_from_torch" begin
    @test_nowarn yolov7_from_torch()
    @test_nowarn yolov7_from_torch(nc=2, class_names=["car", "ball"])
    
    @test_throws SystemError yolov7_from_torch(name="yolo test", pickle_path="")
end
