# Oracle: BLP.Numerics against the old package, bit for bit: units, exact sums (host and
# atomic add), Philox streams and samplers, stream identity, the special functions, the
# Catmull-Rom stencil, the log-linear lookup and the trilinear blend of the samplers.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))

const OS = BlackLightPlus.Simulation
const C = BlackLightPlus.Constants

# Old device bodies, evaluated from their files.
const DEV = scratch_module(:OldDevice)
Core.eval(DEV, :(const _MCM = $OM))
Core.eval(DEV, :(const _XSLOT = $(OM.XSUM_NSLOT)))
Core.eval(DEV, :(using BlackLightPlus.Simulation: units))
load_old!(DEV, "montecarlo/mc_gpu.jl", [:_xsum_atomic_add!])
load_old!(DEV, "gpu_image.jl", [:_gimg_corner, :_gimg_sample, :_unit_fields])
load_old!(DEV, "gpu_amr.jl", [:_gamr_corner, :_gamr_lerp])

bitsof(x) = reinterpret(UInt64, collect(Float64, x))

# Trilinear samplers composed from the new primitives, each with its caller's index map.
corner5(blk, i, j, k, mb) = SVector{8,Float64}(ntuple(f -> Float64(@inbounds blk[f, i, j, k, mb]), Val(8)))
corner4(g, i, j, k) = SVector{8,Float64}(ntuple(f -> Float64(@inbounds g[f, i, j, k]), Val(8)))
cornerA(d, i, j, k, mb) = SVector{8,Float64}(ntuple(p -> Float64(@inbounds d[i, j, k, mb, p]), Val(8)))

function block_lerp(corner, ni, nj, nk, gx, gy, gz)
    il, d1 = NN.corner_weights(gx, ni)
    jl, d2 = NN.corner_weights(gy, nj)
    kl, d3 = NN.corner_weights(gz, nk)
    i1 = il + 1; j1 = jl + 1; k1 = kl + 1; i2 = i1 + 1; j2 = j1 + 1; k2 = k1 + 1
    c = (corner(i1, j1, k1), corner(i2, j1, k1), corner(i1, j2, k1), corner(i2, j2, k1),
         corner(i1, j1, k2), corner(i2, j1, k2), corner(i1, j2, k2), corner(i2, j2, k2))
    return NN.lerp8(c, d1, d2, d3)
end

function uniform_lerp(grid, ng, xo, inv_dx, x, y, z)
    fx = (x - xo) * inv_dx; fy = (y - xo) * inv_dx; fz = (z - xo) * inv_dx
    (fx < 0.0 || fy < 0.0 || fz < 0.0 ||
     fx > ng - 1 || fy > ng - 1 || fz > ng - 1) && return zero(SVector{8,Float64})
    i0 = min(unsafe_trunc(Int, fx), ng - 2)
    j0 = min(unsafe_trunc(Int, fy), ng - 2)
    k0 = min(unsafe_trunc(Int, fz), ng - 2)
    il = i0 + 1; jl = j0 + 1; kl = k0 + 1; ih = il + 1; jh = jl + 1; kh = kl + 1
    c = (corner4(grid, il, jl, kl), corner4(grid, ih, jl, kl), corner4(grid, il, jh, kl),
         corner4(grid, ih, jh, kl), corner4(grid, il, jl, kh), corner4(grid, ih, jl, kh),
         corner4(grid, il, jh, kh), corner4(grid, ih, jh, kh))
    return NN.lerp8(c, fx - i0, fy - j0, fz - k0)
end

@testset "oracle: Numerics" begin
    rng = Xoshiro(161020)

    @testset "units" begin
        nbad = 0
        for _ in 1:2000
            m = exp(rand(rng) * 30.0 + 60.0); s = exp(randn(rng) * 10.0)
            n = NN.Units(m, s); o = OS.units(m, s); f = DEV._unit_fields(m, s)
            nbad += !((n.L_unit, n.T_unit, n.dens_unit, n.Ne_unit, n.B_unit) ===
                      (o.L_unit, o.T_unit, o.dens_unit, o.Ne_unit, o.B_unit) &&
                      n.e_unit === f.e_unit && n.B_unit === f.b_unit && n.dens_unit === f.d_unit)
        end
        @test nbad == 0
        # a Dual mass scale: the values and the partials of the old unit fields
        D = ForwardDiff.Dual{Nothing}(1.0e24, 1.0)
        n = NN.Units(6.2e9 * C.MSUN, D); f = DEV._unit_fields(6.2e9 * C.MSUN, D)
        @test n.dens_unit === f.d_unit && n.e_unit === f.e_unit && n.B_unit === f.b_unit
    end

    @testset "exact sums" begin
        xs = vcat(exp.(50.0 .* randn(rng, 20_000)) .* rand(rng, (-1.0, 1.0), 20_000),
                  ldexp.(randn(rng, 5000), rand(rng, -1074:1023)),
                  [0.0, -0.0, 5.0e-324, -5.0e-324, floatmax(), -floatmax(), Inf, -Inf, NaN,
                   nextfloat(0.0), prevfloat(floatmax()), 2.0^53, 1.0])
        @test all(same(NN.xsum_parts(x), OM.xsum_parts(x)) for x in xs)
        an = NN.xsum_zeros(4); ao = OM.xsum_zeros(4); aa = NN.xsum_zeros(4); ad = NN.xsum_zeros(4)
        for (k, x) in enumerate(xs)
            c = 1 + k % 4
            NN.xsum_add!(an, x, c); OM.xsum_add!(ao, x, c)
            NN.xsum_atomic_add!(aa, x, c); DEV._xsum_atomic_add!(reshape(ad, NN.XSUM_NSLOT, 4, 1), x, c, 1)
        end
        @test an == ao == aa == ad
        @test bitsof(NN.xsum_round_all(an)) == bitsof(OM.xsum_round_all(ao))
        fin = filter(isfinite, xs)
        a2 = NN.xsum_zeros(1); foreach(x -> NN.xsum_add!(a2, x, 1), fin)
        o2 = OM.xsum_zeros(1); foreach(x -> OM.xsum_add!(o2, x, 1), fin)
        @test NN.xsum_round(a2, 1) === OM.xsum_round(o2, 1)
    end

    @testset "Philox and samplers" begin
        nbad = 0
        for _ in 1:5000
            ctr = (rand(rng, UInt32), rand(rng, UInt32), rand(rng, UInt32), rand(rng, UInt32))
            key = (rand(rng, UInt32), rand(rng, UInt32))
            nbad += !(NN.philox4x32(ctr, key) === OM.philox4x32(ctr, key) &&
                      NN.philox_rand(key, ctr) === OM.philox_rand(key, ctr))
        end
        @test nbad == 0
        @test all(NN.philox_rand(s, i, d) === OM.philox_rand(s, i, d)
                  for s in (0, 7, 0xBEEF, typemax(UInt64)), i in (1, 2^40 + 3), d in (0, 11, 2^33 + 5))
        L = NN.MCStreamLayout(10^6, 6000, 3); Lo = OM.MCStreamLayout(10^6, 6000, 3)
        @test (L.gid_bits, L.event_bits, L.max_layer) == (Lo.gid_bits, Lo.event_bits, Lo.max_layer)
        nbad = 0
        for gid in rand(rng, 1:10^6, 300)
            rn = NN.primary_rng(0xC0FFEE, gid); ro = OM.mc_primary_rng(0xC0FFEE, gid)
            for layer in 0:3
                nbad += !(rn.key === ro.key && rn.ctr === ro.ctr && rn.sid === ro.sid &&
                          NN.stream_id(rn) === OM.mc_stream_id(ro) &&
                          NN.stream_layer(rn) === OM.mc_stream_layer(ro))
                for _ in 1:4
                    u, rn = NN.mc_uniform(rn); v, ro = OM.mc_uniform(ro); nbad += u !== v
                    g, rn = NN.mc_gauss(rn); h, ro = OM.mc_gauss(ro); nbad += g !== h
                    d, rn = NN.sample_direction(rn); e, ro = OM.sample_direction(ro); nbad += d !== e
                    sh = rand(rng, (0.3, 0.5, 1.0, 1.5, 2.5, 7.0))
                    g, rn = NN.standard_gamma(sh, rn); h, ro = OM.standard_gamma(sh, ro); nbad += g !== h
                    df = rand(rng, 1:8)
                    g, rn = NN.chi_square(df, rn); h, ro = OM.chi_square(df, ro); nbad += g !== h
                end
                ev = rand(rng, 0:5999)
                rn = NN.child_rng(rn, ev, L); ro = OM.mc_child_rng(ro, ev, Lo)
            end
        end
        @test nbad == 0
        @test NN.MCRng(42, 7, 3) === NN.MCRng(OM.MCRng(42, 7, 3).key, OM.MCRng(42, 7, 3).ctr)
    end

    @testset "special functions" begin
        xs = vcat(exp.(range(log(1e-6), log(700.0); length = 4000)), -rand(rng, 200) .* 10)
        pos = filter(>(0), xs)
        @test all(NN.bessel_i0(x) === OR.bessel_i0(x) && NN.bessel_i1(x) === OR.bessel_i1(x) for x in xs)
        @test all(NN.bessel_k0(x) === OR.bessel_k0(x) && NN.bessel_k1(x) === OR.bessel_k1(x) &&
                  NN.bessel_k2(x) === OR.bessel_k2(x) for x in pos)
        @test all(NN.harmonic_bridge(a, b, x) === OR.harmonic_bridge(a, b, x)
                  for a in exp.(randn(rng, 40) .* 20), b in exp.(randn(rng, 40) .* 20),
                      x in (0.5, 1.0, 2.0, 3.5))
        @test all(NN.hypergeometric_2f1(k - 1 / 3, k + 1.0, k + 2 / 3, z) ===
                  OR._hypergeometric_2f1(k - 1 / 3, k + 1.0, k + 2 / 3, z)
                  for k in (3.5, 4.0, 5.0, 7.5), z in -exp.(randn(rng, 50) .* 3))
    end

    @testset "Catmull-Rom stencil and log-linear lookup" begin
        @test all(NN.cr_weights(t) === OG._cr_weights(t) for t in rand(rng, 2000))
        d = randn(rng, 10, 9, 8, 7)
        nbad = 0
        for _ in 1:3000
            c = SVector(rand(rng) * 12 - 2, rand(rng) * 11 - 2, rand(rng) * 10 - 2)
            ax = (NN.cr_axis(c[1], -0.5, 1.25, 9), NN.cr_axis(c[2], -0.25, 1.5, 8),
                  NN.cr_axis(c[3], 0.0, 1.0, 7))
            ao = (OG._axis(c[1], -0.5, 1.25, 9), OG._axis(c[2], -0.25, 1.5, 8),
                  OG._axis(c[3], 0.0, 1.0, 7))
            nbad += !same(ax, ao)
            comp = rand(rng, 1:10)
            nbad += NN.cr_sample(d, comp, ax[1][2], ax[2][2], ax[3][2], ax[1][1], ax[2][1], ax[3][1]) !==
                    OG._interp(d, comp, ao[1][2], ao[2][2], ao[3][2], ao[1][1], ao[2][1], ao[3][1])
        end
        @test nbad == 0
        lw = randn(rng, 200) .* 30
        @test all(NN.loglinear_at(nu, lw, log(1e8), 0.1, 200) === OM._mc_weight_at(nu, lw, log(1e8), 0.1, 200)
                  for nu in exp.(log(1e8) .+ (rand(rng, 5000) .* 24 .- 2)))
    end

    @testset "trilinear blend of the samplers" begin
        ni, nj, nk, nmb = 6, 5, 7, 2
        blk = rand(rng, Float32, 8, ni, nj, nk, nmb)              # feature-fastest
        data5 = permutedims(blk, (2, 3, 4, 5, 1))                  # (ni, nj, nk, nmb, 8)
        xl = [SVector(-1.0, 0.5, -2.0), SVector(2.0, 0.5, -2.0)]
        dx = [SVector(0.5, 0.75, 0.25), SVector(0.5, 0.75, 0.25)]
        lo = [xl[m] - dx[m] / 2 for m in 1:nmb]; hi = [lo[m] + dx[m] .* (ni, nj, nk) for m in 1:nmb]
        snap = OS.AthenaKSnapshot(data5, lo, hi, xl, dx, [0, 0], zeros(Int, 3, nmb),
                                  ["dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"],
                                  ni, nj, nk, nmb)
        xlft = reduce(hcat, xl); dxb = reduce(hcat, dx)
        nbad = 0; ninside = 0
        for _ in 1:5000
            p = SVector(rand(rng) * 6.5 - 1.5, rand(rng) * 4, rand(rng) * 2.1 - 2.3)
            old = OS.sample_primitives(snap, SVector(0.0, p[1], p[2], p[3]))
            old.inside || continue
            ninside += 1
            mb = OS.meshblock_index(snap, p[1], p[2], p[3])
            g = (p - xl[mb]) ./ dx[mb]
            new = block_lerp((i, j, k) -> cornerA(data5, i, j, k, mb), ni, nj, nk, g[1], g[2], g[3])
            amr = DEV._gamr_lerp(blk, xlft, dxb, ni, nj, nk, mb, p[1], p[2], p[3])
            gx = (p[1] - xlft[1, mb]) / dxb[1, mb]; gy = (p[2] - xlft[2, mb]) / dxb[2, mb]
            gz = (p[3] - xlft[3, mb]) / dxb[3, mb]
            new5 = block_lerp((i, j, k) -> corner5(blk, i, j, k, mb), ni, nj, nk, gx, gy, gz)
            o = SVector(old.rho, old.uu[1], old.uu[2], old.uu[3], old.eint, old.bb[1], old.bb[2], old.bb[3])
            nbad += !(new === o && new5 === amr && amr === o)
        end
        @test ninside > 1000
        @test nbad == 0
        ng = 9; grid = rand(rng, Float32, 8, ng, ng, ng); xo = -3.0; inv_dx = (ng - 1) / 6.0
        @test all(uniform_lerp(grid, ng, xo, inv_dx, p...) === DEV._gimg_sample(grid, ng, xo, inv_dx, p...)
                  for p in (SVector(rand(rng) * 7 - 3.5, rand(rng) * 7 - 3.5, rand(rng) * 7 - 3.5)
                            for _ in 1:5000))
        # The Z4c resample blends as a(1 - t) + b t: lerp8 is not its arithmetic.
        nx = 6; zdata = rand(rng, Float32, nx, nx, nx, 2, 1)
        zb = OS.AthenaKBin(1, nx, nx, nx, ["a", "b"], [0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0],
                           zdata, 0.0)
        ndiff = 0
        for _ in 1:2000
            i0, j0, k0 = rand(rng, 1:nx-1, 3); t = rand(rng, 3)
            z = OS._sample_stencil(zb, OS._Stencil(1, i0, j0, k0, t...), 1)
            g(i, j, k) = Float64(zdata[i, j, k, 1, 1])
            c = (g(i0, j0, k0), g(i0 + 1, j0, k0), g(i0, j0 + 1, k0), g(i0 + 1, j0 + 1, k0),
                 g(i0, j0, k0 + 1), g(i0 + 1, j0, k0 + 1), g(i0, j0 + 1, k0 + 1), g(i0 + 1, j0 + 1, k0 + 1))
            ndiff += NN.lerp8(c, t...) !== z
        end
        @test ndiff > 0
        @info "Z4c resample vs lerp8: samples with different bits" ndiff total = 2000
    end
end
