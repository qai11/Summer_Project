"""
Title: PSF_photometry.py
Author: Zachary Lane, Quin Aicken Davies
Date: 07/11/2025

Description: Perform PSF photometry
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
from copy import deepcopy

from scipy.optimize import curve_fit

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.time import Time

from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from photutils.aperture import ApertureStats
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from astropy.utils.exceptions import AstropyWarning
#PSF modules
from photutils.psf import extract_stars, EPSFStars, EPSFBuilder, EPSFModel
from photutils.psf import PSFPhotometry, IterativePSFPhotometry

import warnings
from astropy.coordinates import SkyCoord

#%%
