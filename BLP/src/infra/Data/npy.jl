# NumPy `.npy` writer (format version 1.0), with no dependency on NumPy or Python.

"""
    write_npy(path, a::AbstractMatrix{Float64}) -> path

Write `a` to `path` as a NumPy `.npy` file, format version 1.0: the magic string, the
header `{'descr': '<f8', 'fortran_order': False, 'shape': (m, n), }` padded with spaces
and a newline to a multiple of 64 bytes, then the little-endian `Float64` values in C
(row-major) order, so `numpy.load(path)[i-1, j-1] == a[i, j]`.

# References
NumPy Enhancement Proposal 1, "A simple file format for NumPy arrays".
"""
function write_npy(path::AbstractString, a::AbstractMatrix{Float64})
    shape = size(a)
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': ($(shape[1]), $(shape[2])), }"
    pad = 64 - mod(6 + 2 + 2 + length(header) + 1, 64)   # magic + version + length + header + \n
    header = header * repeat(" ", pad) * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93], "NUMPY", UInt8[0x01, 0x00])
        write(io, UInt16(length(header)))
        write(io, header)
        # the column-major bytes of the transpose are the row-major stream
        write(io, permutedims(a))
    end
    return path
end
