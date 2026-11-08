# GPU execution: plan and its evidence

- `PLAN.md`: the work plan (goals with measured baselines, work items as a dependency graph, gates, GPU job budget, risks, decisions for the user).
- `inputs/profile.md`: the A100 profile the design rests on.
- `inputs/map_*.md`: maps of the execution layer, the Monte Carlo programs, the march, the compiler and bit portability, the outputs, and the literature.
- `inputs/design_*.md`: the three executor designs judged against each other and the design of the other GPU items.

The design standard itself is `docs/src/design/gpu_execution.md` on branch `gpu-wavefront` until it lands.
