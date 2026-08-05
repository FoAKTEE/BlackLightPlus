# Outputs API

The `Outputs` module writes images and computes interferometric observables.

## Image output

```@docs
write_npy
```

## Interferometry

Visibility amplitudes from an image, following AART
[CardenasAvendano2023](@citep). `radon_cut` takes a projection through the
image; `visamp` computes the visibility amplitude along a baseline;
`visamp_minima` locates the minima whose spacing encodes the photon-ring
diameter.

```@docs
radon_cut
visamp
visamp_minima
tukey_window
MUAS_TO_RAD
```
