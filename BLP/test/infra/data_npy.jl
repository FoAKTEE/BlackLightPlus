# The `.npy` writer: magic, version, a 64-byte aligned header, and the row-major payload.
using Test, BLP
using BLP.Data: write_npy

@testset "write_npy: byte-correct C-order .npy" begin
    for a in (reshape(collect(1.0:15.0), 5, 3) .+ 0.25, fill(-0.0, 1, 1), rand(64, 7))
        path = tempname() * ".npy"
        @test write_npy(path, a) == path
        bytes = read(path)
        @test bytes[1:6] == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]      # \x93NUMPY
        @test bytes[7] == 0x01 && bytes[8] == 0x00                          # version 1.0
        hlen = Int(reinterpret(UInt16, bytes[9:10])[1])
        @test (10 + hlen) % 64 == 0 && bytes[10 + hlen] == UInt8('\n')
        hdr = String(copy(bytes[11:10 + hlen]))
        @test occursin("'shape': ($(size(a, 1)), $(size(a, 2)))", hdr)
        @test occursin("'descr': '<f8'", hdr) && occursin("'fortran_order': False", hdr)
        payload = reinterpret(Float64, bytes[11 + hlen:end])
        @test length(payload) == length(a)
        @test reinterpret(UInt64, payload) ==
              reinterpret(UInt64, [a[i, j] for i in 1:size(a, 1) for j in 1:size(a, 2)])
        rm(path; force = true)
    end
end
