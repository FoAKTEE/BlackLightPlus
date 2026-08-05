"""
    Constants

CGS physical constants (values as in blacklight `constants.hpp` and ipole).
"""
module Constants

export CL, EE, ME, MP, KB, HPL, GNEWT, MSUN, SQRT2, SIGMA_THOMSON

const CL    = 2.99792458e10      # speed of light [cm/s]
const EE    = 4.8032e-10         # electron charge [esu]
const ME    = 9.1094e-28         # electron mass [g]
const MP    = 1.6726e-24         # proton mass [g]
const KB    = 1.3807e-16         # Boltzmann [erg/K]
const HPL   = 6.6261e-27         # Planck [erg s]
const GNEWT = 6.6743e-8          # Newton G [cgs]
const MSUN  = 1.989e33           # solar mass [g]
const SQRT2 = sqrt(2.0)
const SIGMA_THOMSON = 0.665245873e-24  # Thomson cross section [cm^2] (Monte Carlo Compton)

end # module
