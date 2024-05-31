# BSD 3-Clause License
# 
# Copyright (c) 2024 Shubh Agrawal
# All rights reserved.

# writing down line properties

# Paper References:


from astropy import units as u, constants as c
import utils
from obj import AttrDict, cosmo
import numpy as np

CII = AttrDict()
CII.l = 157.74 * u.micron # Cooksy et al. 1986, [CII]
CII.nu = CII.l.to(u.Hz, equivalencies=u.spectral())

def Inu_proposal(z, L0=8.3 * 10**6 * u.Lsun * u.yr / u.Msun, nu_emit=CII.nu):
    """
    Calculate the specific intensity of a source at redshift z, using Juzz's proposal formalization.

    Parameters:
    z (float or array-like): Redshift value.
    L0 (float): ratio of line intensity to LIR.
    nu_emit (float, optional): Rest-frame frequency of the emission line in Hz. 
            Defaults to nuCII.

    Returns (Quantity): Specific intensity in Jy/sr.
    """
    sfrd = utils.MD_sfrd(z)
    # TODO: try Ryan's various versions for SFRD --> LCII
    # TODO: L0 definition does not make sense: taken after some computation from DeLooze?
    if not hasattr(sfrd, "unit"):
        sfrd *= u.Msun / u.yr / u.Mpc ** 3
    if not hasattr(nu_emit, "unit"):
        nu_emit *= u.Hz
    eps = L0 * sfrd # TODO: stopgap for right now
    Ivals = c.c * eps / (4 * np.pi * u.sr * cosmo.H(z) * nu_emit)
    # TODO: (4 * np.pi * cosmo.luminosity_distance(z).to(u.m).value ** 2)
    return Ivals.to(u.Jy / u.sr)

def Inu_DeLooze(z, m=1.01, b=-6.99, nu_emit=CII.nu):
    """
    Calculate the specific intensity of a source at redshift z, using DeLooze+2014 formalization.
    
    Parameters:
    z (float or array-like): Redshift value.
    m (float): Slope of the SFRD-LCII relation.
    b (float): Intercept of the SFRD-LCII relation.
    
    Returns (Quantity): Specific intensity in Jy/sr.
    """
    sfrd = utils.MD_sfrd(z)
    if hasattr(sfrd, "unit"):
        sfrd = sfrd.to(u.Msun / u.yr / u.Mpc ** 3).value
    Lval = np.power(10, (np.log10(sfrd) - b) / m) * u.Lsun / u.Mpc ** 3
    Ivals = Lval * c.c / (4 * np.pi * u.sr * cosmo.H(z) * nu_emit) 
    # TODO: luminosity distance?
    return Ivals.to(u.Jy / u.sr)

def Inu_Spinoglio(z, A=0.89, B=2.67):
    """
    Calculate the specific intensity of a source at redshift z, using Spinoglio+2012 formalization,
    accounting for errata from 2014.
    
    Parameters:
    z (float or array-like): Redshift value.
    m (float): Slope of the SFRD-LCII relation.
    b (float): Intercept of the SFRD-LCII relation.
    
    Returns (Quantity): Specific intensity in Jy/sr.
    """
    sfrd = utils.MD_sfrd(z)
    if hasattr(sfrd, "unit"):
        sfrd = sfrd.to(u.Msun / u.yr / u.Mpc ** 3).value
    L_ir = sfrd * 1e10 * u.Lsun # from simIM code (:missing the per volume factor:)
    L_ir = L_ir.to(10 ** 41 * u.erg / u.s).value # units Spinoglio uses
    Lval = np.power(10, A * np.log10(L_ir) - B) * 10 ** 41 * u.erg / u.s
    Lval /= u.Mpc ** 3 # add back the per volume factor
    
    Ivals = Lval * c.c / (4 * np.pi * u.sr * cosmo.H(z) * CII.nu)
    # TODO: luminosity distance?
    return Ivals.to(u.Jy / u.sr)

CII.Inu = Inu_proposal # Inu(z) using SFRD