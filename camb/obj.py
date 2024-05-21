# BSD 3-Clause License
# 
# Copyright (c) 2024 Shubh Agrawal
# All rights reserved.

# maybe useful python objects

# the cosmology used throughout this analysis
from astropy.cosmology import Planck18 as cosmo

class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value
        
    def __delattr__(self, key):
        del self[key]