# The register caps of the products' programs (`Execution.register_cap`): the thin Monte
# Carlo primary and the production crossing program compile to at most 168 registers; the
# scattering primaries, the layers and the ninth-order crossing program keep the
# compiler's limit. A capped production render and an uncapped precise one on a pool (the
# GPUs under BLP_TEST_DEVICES=cuda) agree with threads.
using Test, BlackLightPlus
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare, precise
using BlackLightPlus.Products: MCPrimary, MCLayer, MCTransport, CrossingProgram, render_crossings
using BlackLightPlus.Spacetime: KerrSchildCartesian
using BlackLightPlus.Matter: EquatorialDisk
using BlackLightPlus.Execution: PoolExecutor, CameraRays, register_cap
const Decks = Main.Decks

transport(scatter) = MCTransport(nothing, nothing, nothing, nothing, Val(true), Val(scatter))
agree(a, b) = Decks.CROSS_HARDWARE ? maximum(abs.(a .- b)) <= 1.0e-9 * maximum(abs.(b)) :
              Decks.bits(a) == Decks.bits(b)

@testset "register caps of the products" begin
    @test register_cap(MCPrimary(nothing, transport(false), nothing, UInt32(1))) == 168
    @test register_cap(MCPrimary(nothing, transport(true), nothing, UInt32(1))) === nothing
    @test register_cap(MCLayer(nothing, transport(true), nothing)) === nothing
    cam = CameraConfig(inclination_deg = 80.0, fov = 24.02, resolution = 12)
    d = EquatorialDisk(0.99)
    for (cfg, cap) in ((RunConfig(camera = cam), 168), (precise(RunConfig(camera = cam)), nothing))
        p = prepare(cfg; spacetime = KerrSchildCartesian(0.99), matter = nothing)
        prog = CrossingProgram(p.geodesic_spacetime, p.geodesic_budget,
                               CameraRays(p.spacetime, p.frame, p.pixels), d, (0, 1, 2))
        @test register_cap(prog) === cap
        r = render_crossings(p, d; backend = :cpu)
        q = render_crossings(p, d; backend = PoolExecutor(Decks.devices(2); chunk = 37))
        @test all(agree(q.Ib[b], r.Ib[b]) for b in 1:3) && agree(q.steps, r.steps)
    end
end
