using Test

@testset verbose=true "YOLOv7.jl" begin

    @testset "Yolo" begin
      include("yolo.jl")
    end

end