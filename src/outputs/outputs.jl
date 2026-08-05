"""
    Outputs

Image writers and interferometry (blacklight `outputs/` heritage): a
zero-dependency NumPy `.npy` (format v1.0) writer, mirroring blacklight's
in-house `npy.cpp`, whose arrays load directly with `numpy.load`.
"""
module Outputs

export write_npy,
       radon_cut, visamp, visamp_minima, tukey_window, MUAS_TO_RAD

"Write a Float64 matrix as NumPy .npy v1.0 with shape `size(a)` (C element order)."
function write_npy(path::AbstractString, a::AbstractMatrix{Float64})
    shape = size(a)
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': ($(shape[1]), $(shape[2])), }"
    pad = 64 - mod(6 + 2 + 2 + length(header) + 1, 64)   # magic+ver+hlen+header+\n -> 64-aligned
    header = header * repeat(" ", pad) * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93], "NUMPY", UInt8[0x01, 0x00])
        write(io, UInt16(length(header)))
        write(io, header)
        # C order: the column-major bytes of `permutedims(a)` are exactly the
        # row-major stream, written in one bulk call.
        write(io, permutedims(a))
    end
    return path
end

include("interferometry.jl")

end # module
