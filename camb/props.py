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
TIM.SW.NEI = 12.41e7 * u.Jy / (u.s ** .5)
TIM.SW.num_dets = 64
TIM.SW.dnu = 4.4 * u.GHz

TIM.LW.min = 317 * u.micron
TIM.LW.max = 420 * u.micron
TIM.LW.NEI = 6.81e7 * u.Jy / (u.s ** .5)
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
LADUMA.maxbase = 7700 * u.m
LADUMA.Tsys = 28 * u.K # TODO(shubh): this is nu dependent, but approximating for now
# https://skaafrica.atlassian.net/servicedesk/customer/portal/1/article/277315585

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
FYST.Dap = 9 * u.m
FYST.nei = 4.84e4 * u.Jy * (u.s ** .5) / u.sr
FYST.nfeed = 1
FYST.Dnu = 40 * u.GHz
FYST.dnu = 400 * u.MHz