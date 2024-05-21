# BSD 3-Clause License
# 
# Copyright (c) 2024 Shubh Agrawal, Justin Bracks
# All rights reserved.

import numpy as np
from obj import cosmo
from astropy import constants as c, units as u
from lines import CII

h = cosmo.h

def l2z_CII(l_obs, l_emit=CII.l):
    """
    Convert observed wavelength to redshift for CII emission line.

    Parameters:
    l_obs (float): The observed wavelength in micron.
    l_emit (float, optional): The rest-frame wavelength of the CII emission line in micron. 
            Defaults to lCII.

    Returns: (float) The redshift corresponding to the observed wavelength.
    """
    # if l_obs does not have units, add them
    if not hasattr(l_obs, "unit"):
        l_obs = l_obs * u.micron
    assert l_obs > l_emit, "Observed wavelength must be greater than rest-frame wavelength."
    return (l_obs - l_emit) / l_emit

def num_log_steps(x_start, x_stop, dlnx):
    """
    Calculate the number of steps needed to go from x_start to x_stop with a step size of dlnx.

    Parameters:
    x_start (float): The starting value.
    x_stop (float): The final value.
    dlnx (float): The step size in log space.
    
    Returns: (float) The number of steps needed to go from x_start to x_stop.
    """
    return np.log(x_stop / x_start) / np.log(1 + dlnx)

def transverse_scale(alpha, z_range, lilh=True):
    """
    Calculate the comoving transverse scale, using the provided 
    angular scale and redshift values

    Parameters:
    alpha (Quantity or float): Angular scale in radians.
    z_range (array-like or float): Redshift values.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Comoving transverse scale in Mpc.
    """
    if hasattr(alpha, "unit"):
        alpha = alpha.to(u.rad).value
    transLs = cosmo.comoving_transverse_distance(z_range) * alpha
    return transLs * h if lilh else transLs

def los_extent(z_min, z_max, lilh=True):
    """
    Calculate the comoving line-of-sight extent between two redshifts.

    Parameters:
    z_min (float): The lower redshift value.
    z_max (float): The upper redshift value.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Comoving line-of-sight extent in Mpc.
    """
    ext = cosmo.comoving_distance(z_max) - cosmo.comoving_distance(z_min)
    return ext * h if lilh else ext

def dnu2dr(dnu, z, lilh=True, nu_emit=CII.nu):
    """
    Calculate the comoving radial distance corresponding to a (observed) frequency interval.

    Parameters:
    dnu (Quantity or float): Frequency interval in Hz.
    z (float): Redshift value.
    lilh (bool, optional): If True, use little-h units. Default is True.
    nu_emit (Quantity or float, optional): Rest-frame frequency of the emission line in Hz.
            Defaults to nuCII.

    Returns (Quantity): Comoving radial distance in Mpc.
    """
    if hasattr(dnu, "unit"):
        dnu = dnu.to(u.Hz).value
    if hasattr(nu_emit, "unit"):
        nu_emit = nu_emit.to(u.Hz).value
    nu_obs = nu_emit / (1 + z)
    dR = cosmo.comoving_distance(z) - cosmo.comoving_distance(nu_emit / (nu_obs + dnu) - 1)
    return dR * h if lilh else dR

def volume_cube(z_min, z_max, d_omega, lilh=True):
    """
    Calculate the cosmological volume enclosed between z_min, z_max, for 
    a given solid angle d_omega.

    Parameters:
    z_min (float): Redshift at the near face.
    z_max (float): Redshift at the farther face.
    d_omega (float): Solid angle in steradians.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Survey volume in Mpc^3.
    """
    volume_shell = cosmo.comoving_volume(z_max) - cosmo.comoving_volume(z_min)
    sr_ratio = d_omega / (4 * np.pi * u.sr)
    return volume_shell * sr_ratio * h**3 if lilh else volume_shell * sr_ratio

def volume_scan(z_min, z_max, d_az, d_el, lilh=True):
    """
    Calculate the cosmological volume enclosed between z_min, z_max, for 
    a given angular resolution.

    Parameters:
    z_min (float): Redshift at the near face.
    z_max (float): Redshift at the farther face.
    d_az (float): Azimuthal edge in radians.
    d_el (float): Elevation edge in radians.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Survey volume in Mpc^3.
    """
    return volume_cube(z_min, z_max, d_az * d_el, lilh)

def area_scan(z, az, el, lilh=True):
    """
    Calculate the transverse area enclosed by az and el at redshift z.

    Parameters:
    z (float or array-like) : redshift value(s)
    az (float) : azimuthal scale/range in radians.
    el (float) : elevation scale/range in radians.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (float): The transverse area enclosed by az and el at redshift z.
    """
    return transverse_scale(az, z, lilh) * transverse_scale(el, z, lilh) # TODO: this is wrong in Juzz's code, extra h**2?

def num_modes(del_ln_k, k, volume): # TODO: unused
    """
    Calculate the number of modes in a given volume.

    Parameters:
    del_ln_k (float): The logarithmic step size in k-space.
    k (array-like or float): The wavenumber values.
    volume (Quantity): survey volume in Mpc^3.

    Returns (float): The number of modes in the given volume.
    """
    # return volume * np.log(k[-1] / k[0]) / np.log(1 + del_ln_k)
    return 4 * np.pi * volume * del_ln_k * k ** 3 / (2 * np.pi) ** 3 #TODO: Check this


def noise_per_cell(z, nei, num_spax, num_dets, time, dAz, dEl, dnu, lilh=True):
    """
    Calculate the noise per cell in the survey.

    Parameters:
    z (float): Redshift value.
    nei (float): Noise equivalent intensity.
    num_spax (float): Number of spaxels.
    time (float): Integration time in seconds.
    dAz (float): Azimuthal resolution in radians.
    dEl (float): Elevation resolution in radians.
    dnu (float): Frequency resolution in Hz.
    lilh (bool, optional): If True, use little-h units. Default is True.

    Returns (Quantity): Noise per cell.
    """
    dR = dnu2dr(dnu, z, lilh)
    cell_area = area_scan(z, dAz, dEl, lilh) # TODO: this is wrong in Juzz's code, extra h**2?
    cell_vol = cell_area * dR
    cell_time = time / num_spax * num_dets
    return (nei * np.sqrt(cell_vol / cell_time)) ** 2 #TODO: why square?

def MD_sfrd(z):
    """
    Calculate the star formation rate density at redshift z, using the Madau-Dickinson model.
    EQ15 from https://arxiv.org/pdf/1403.0007

    Parameters:
    z (float or array-like): Redshift value.

    Returns (float): Star formation rate density in Msun/yr/Mpc^3.
    """
    val = 0.015 * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)
    return val * u.Msun / u.yr / u.Mpc ** 3

def calc_k_modes(Da, da):
    """
    Calculate the k modes for a given extent and resolution element.
    
    Parameters:
    - Da (float): extent in a direction.
    - da (float): resolution element in the same direction.
    
    Returns:
    - k_modes (ndarray): The calculated k modes.
    """
    return 2 * np.pi * np.fft.fftfreq(int(Da.value / da.value), da.value)
        

class zBin:
    def __init__(self, zfront, zback):
        """
        initialize a redshift bin with the front and back redshifts.
        """
        self.front = zfront
        self.back = zback
        self.center = (zfront + zback) / 2
        
    def vCube(self, dOmega, lilh = True):
        """
        calculate the volume of the redshift bin for a given solid angle.
        """
        return volume_cube(self.front, self.back, dOmega, lilh)
    def vScan(self, dAz, dEl, lilh = True):
        """
        calculate the volume of the redshift bin for a given angular resolution.
        """
        return volume_scan(self.front, self.back, dAz, dEl, lilh)
    def SFRD(self):
        """
        calculate the star formation rate density in the redshift bin.
        """
        return MD_sfrd(self.center)
    def LoSmax(self, lilh = True):
        """
        calculate the maximum line-of-sight extent of the redshift bin.
        """
        return los_extent(self.front, self.back, lilh)
    def LoSmin(self, dnu):
        """
        calculate the minimum line-of-sight extent (resolution) of the redshift bin.
        """
        return dnu2dr(dnu, self.center)
    def transScaleFront(self, angle):
        """
        calculate the transverse scale at the front of the redshift bin.
        """
        return transverse_scale(angle, self.front)
    def transScaleBack(self, angle):
        """
        calculate the transverse scale at the back of the redshift bin.
        """
        return transverse_scale(angle, self.back)
    def transScale(self, angle):
        """
        calculate the transverse scale at the center of the redshift bin.
        """
        return transverse_scale(angle, self.center)
    
class Instrument:
    def __init__(self, nei, fwhm, dnu, num_dets):
        """
        initialize an instrument with the noise equivalent intensity, FWHM, 
        frequency resolution and number of detectors.
        """
        self.NEI = nei
        self.FWHM = fwhm
        self.dnu = dnu
        self.dOmega = fwhm ** 2
        self.num_dets = num_dets
        
class Survey:
    def __init__(self, z_info, ins_info, AZ, EL, lilh=True):
        self.ins = ins_info
        self.z = z_info
        self.az = AZ
        self.el = EL
        self.dAz = self.dEl = ins_info.FWHM
        self.vol = z_info.vScan(AZ, EL, lilh)
        self.LoS = z_info.LoSmax(lilh)
        self.dLoS = z_info.LoSmin(ins_info.dnu)
        self.vol_spax = z_info.vCube(ins_info.dOmega)
        self.num_spax = (AZ * EL / ins_info.FWHM ** 2).decompose() #TODO: add unit hasattr+checker
    def cell_noise(self, time, lilh=True):
        """
        calculate the noise per cell in the survey.
        """
        return noise_per_cell(self.z.center, self.ins.NEI, self.num_spax, 
                            self.ins.num_dets, time, self.dAz, self.dEl,
                            self.ins.dnu, lilh)
    def k_modes(self):
        """
        calculate the k-modes in the survey.
        """
        Dx = self.z.transScale(self.az)
        Dy = self.z.transScale(self.el)
        Dz = self.LoS

        dx = transverse_scale(self.ins.FWHM, self.z.center)
        dy = transverse_scale(self.ins.FWHM, self.z.center)
        dz = self.dLoS
        
        kx, ky, kz = np.meshgrid(calc_k_modes(Dx, dx), calc_k_modes(Dy, dy), 
                                calc_k_modes(Dz, dz), indexing='ij')
        
        return np.stack((kx, ky, kz))
