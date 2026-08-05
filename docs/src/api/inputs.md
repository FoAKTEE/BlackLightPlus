# Configuration API

The `Inputs` module holds the configuration objects passed to every render
entry point. `RunConfig` is the top-level container; it nests a `CameraConfig`
describing the observer and a `PlasmaConfig` describing the electron model.

```@docs
BlackLightPlus.Inputs
RunConfig
CameraConfig
PlasmaConfig
```
