# athdf_to_bin.jl — write AthenaK `.athdf` dumps (datasets uov = dens, velx, vely, velz,
# eint and B = bcc1..3) as AthenaK binary dumps (version 1.1, 8-byte locations, 4-byte
# variables, the input deck of the dump as the parameter block), the layout Blacklight's
# `athenak` reader and BlackLightPlus's `read_athenak_bin` share:
# per block Int32[ois, oie, ojs, oje, oks, oke, lx1, lx2, lx3, level], Float64[6] face
# bounds, Float32[nvar, nz, ny, nx] cell data. The time is the dump's.
# Usage: julia --project=. figures/kpolaris/athdf_to_bin.jl <out_dir> <athdf>...
using HDF5

function convert(athdf::AbstractString, out::AbstractString)
    h5open(athdf, "r") do h
        uov = read(h["uov"]); B = read(h["B"])              # (ni, nj, nk, nmb, nvar)
        names = String.(read_attribute(h, "VariableNames"))
        t = Float64(read_attribute(h, "Time"))
        x1f = read(h["x1f"]); x2f = read(h["x2f"]); x3f = read(h["x3f"])   # (nfaces, nmb)
        lev = read(h["Levels"]); ll = read(h["LogicalLocations"])          # (nmb), (3, nmb)
        ni, nj, nk, nmb, nh = size(uov)
        nb = size(B, 5)
        data = cat(uov, B; dims = 5)
        # the dump's input deck (the `Header` attribute), which carries <mhd> gamma
        hdr = String.(read_attribute(h, "Header"))
        params = join(hdr, "\n") * "\n"
        open(out, "w") do io
            print(io, "Athena binary output version=1.1\n  size of preheader=5\n  time=$t\n",
                  "  cycle=1\n  size of location=8\n  size of variable=4\n",
                  "  number of variables=", nh + nb, "\n",
                  "  variables:  ", join(names, "  "), "  \n",
                  "  header offset=", length(params), "\n", params)
            for b in 1:nmb
                write(io, Int32[0, ni - 1, 0, nj - 1, 0, nk - 1, ll[1, b], ll[2, b], ll[3, b], lev[b]])
                write(io, Float64[x1f[1, b], x1f[end, b], x2f[1, b], x2f[end, b], x3f[1, b], x3f[end, b]])
                write(io, Float32.(data[:, :, :, b, :]))
            end
        end
        println(out, "  t=", t, "  blocks=", nmb, "  vars=", join(names, ","))
    end
end

outdir = ARGS[1]
mkpath(outdir)
for f in ARGS[2:end]
    num = match(r"\.(\d{5})\.athdf$", f).captures[1]
    convert(f, joinpath(outdir, "torus_$num.bin"))
end
