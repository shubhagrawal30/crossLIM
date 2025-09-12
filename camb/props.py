# BSD 3-Clause License
# 
# Copyright (c) 2024 Shubh Agrawal
# All rights reserved.

# writing down survey properties

from astropy import units as u, constants as c
import utils
from obj import AttrDict
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import lines as l

TIM = AttrDict()
TIM.time = (200*u.hr).to(u.s)
TIM.window = True
TIM.useshot = True
TIM.mirror = 2 * u.m
TIM.Daz = 0.2 * u.deg
TIM.Del = 1 * u.deg
TIM.line = 157.74 * u.micron # Cooksy et al. 1986, [CII]


TIM.SW = AttrDict()
TIM.LW = AttrDict()

TIM.SW.min = 240 * u.micron
TIM.SW.max = 317 * u.micron
TIM.SW.NEI = 12.41e7 * u.Jy * (u.s ** .5)
TIM.SW.num_dets = 64
TIM.SW.dnu = 4.4 * u.GHz

TIM.LW.min = 317 * u.micron
TIM.LW.max = 420 * u.micron
TIM.LW.NEI = 6.81e7 * u.Jy * (u.s ** .5)
TIM.LW.num_dets = 51
TIM.LW.dnu = 3.3 * u.GHz

for band in ['SW', 'LW']:
    TIM[band].cen = (TIM[band].min + TIM[band].max) / 2
    TIM[band].FWHM = (1.22 * TIM[band].cen / TIM.mirror).to("").value * u.rad
    TIM[band].zmin = utils.l2z_CII(TIM[band].min)
    TIM[band].zcen = utils.l2z_CII(TIM[band].cen)
    TIM[band].zmax = utils.l2z_CII(TIM[band].max)
    TIM[band].ins = utils.Instrument(TIM[band].NEI, TIM[band].FWHM, \
        TIM[band].dnu, TIM[band].num_dets)

AstroDeep = AttrDict()
# TIM bins
AstroDeep.n_gals = np.array([0.005, 0.002, 0.0019713, 0.000836]) / cosmo.h ** 3

Euclid = AttrDict()
# TIM bins
Euclid.n_gals = np.array([0.0144, 0.01077, 0.0081,0.0056]) #little h included in these numbers.


# Meerkat LADUMA
LADUMA = AttrDict()
LADUMA.time = 96 * u.hr
LADUMA.ndish = 64
LADUMA.dish = 13.5 * u.m
LADUMA.minbase = 29 * u.m
# LADUMA.maxbase = 7700 * u.m
LADUMA.maxbase = 2000 * u.m
LADUMA.Tsys = 28 * u.K # TODO(shubh): this is nu dependent, but approximating for now
# https://skaafrica.atlassian.net/servicedesk/customer/portal/1/article/277315585
LADUMA.dnu = 208.984 * u.kHz # channel width
LADUMA.line = l.HI
LADUMA.zmin = 0.0
LADUMA.zmax = 1.6
LADUMA.zcen = (LADUMA.zmin + LADUMA.zmax) / 2
LADUMA.obslam = l.HI.l * (1 + LADUMA.zcen)
LADUMA.FWHM = ((1.22 * LADUMA.obslam) / LADUMA.maxbase).to("").value * u.rad
LADUMA.ins = utils.Instrument(0, LADUMA.FWHM, LADUMA.dnu, 0)
LADUMA.min = LADUMA.line.l * (1 + LADUMA.zmin)
LADUMA.max = LADUMA.line.l * (1 + LADUMA.zmax)
LADUMA.Adish = np.pi * (LADUMA.dish / 2) ** 2


TIME = AttrDict()
TIME.time = 3000 * u.hr
TIME.Dap = 12 * u.m
TIME.beam = 0.43 * u.arcmin
TIME.nei = 5e6 * u.Jy * (u.s ** .5) / u.sr
TIME.nfeed = 32
TIME.R = 105
TIME.nu = 250 * u.GHz

FYST = AttrDict()
FYST.time = 200 * u.hr
FYST.Dap = 6 * u.m
FYST.NEI = 1.8e4 * u.Jy * (u.s ** .5) / u.sr # https://articles.adsabs.harvard.edu/pdf/2023pcsf.conf..352N
# band is around 250 GHz
FYST.dnu = 2.8 * u.GHz
FYST.R = 100
FYST.nbeams = 6912

FYST.CO43 = AttrDict()
FYST.CO43.zmin = 0.5
FYST.CO43.zmax = 1.0
FYST.CO43.zcen = (FYST.CO43.zmin + FYST.CO43.zmax) / 2
FYST.CO43.line = l.CO43
FYST.CO43.obslam = l.CO43.l * (1 + FYST.CO43.zcen)
FYST.CO43.FWHM = ((1.22 * FYST.CO43.obslam) / FYST.Dap).to("").value * u.rad
FYST.CO43.ins = utils.Instrument(FYST.NEI, FYST.CO43.FWHM, FYST.dnu, FYST.nbeams)
FYST.CO43.min = FYST.CO43.line.l * (1 + FYST.CO43.zmin)
FYST.CO43.max = FYST.CO43.line.l * (1 + FYST.CO43.zmax)


FYST.CO54 = AttrDict()
FYST.CO54.zmin = 1.0
FYST.CO54.zmax = 1.6
FYST.CO54.zcen = (FYST.CO54.zmin + FYST.CO54.zmax) / 2
FYST.CO54.line = l.CO54
FYST.CO54.obslam = l.CO54.l * (1 + FYST.CO54.zcen)
FYST.CO54.FWHM = ((1.22 * FYST.CO54.obslam) / FYST.Dap).to("").value * u.rad
FYST.CO54.ins = utils.Instrument(FYST.NEI, FYST.CO54.FWHM, FYST.dnu, FYST.nbeams)
FYST.CO54.min = FYST.CO54.line.l * (1 + FYST.CO54.zmin)
FYST.CO54.max = FYST.CO54.line.l * (1 + FYST.CO54.zmax)


# FYST.nei = 4.84e4 * u.Jy * (u.s ** .5) / u.sr
# FYST.nfeed = 1
# FYST.Dnu = 40 * u.GHz
# FYST.dnu = 400 * u.MHz
# FYST.R = 100
# FYST.min = 210 * u.GHz
# FYST.max = 420 * u.GHz