"""
Cross-correlation power spectrum analysis for CII, HI, and CO line emissions.

This module computes power spectra and cross-correlation coefficients for 
different line emission tracers using TNG100 simulation data.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u, constants as con
from multiprocessing import Pool
import os, sys, gc, pickle
from itertools import product
from pathlib import Path

try:
    import psutil  # For memory monitoring
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available - memory monitoring disabled")

import simim.siminterface as sim
from simim import constants as sc
from simim.galprops import (
    prop_behroozi_sfr, prop_delooze_cii, 
    prop_li_hi, prop_co_sled, prop_mass_hi,
    prop_halpha, prop_galaxy_survey
)


# Configuration constants
ZINDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
RANDOM_SEED = 1511
GRID_RESOLUTION = 450  # pixels per box edge
Z_MODELING = [0.644, 0.888, 1.173, 1.499][ZINDEX]
# Z_MODELING = [.32, .44][ZINDEX]
CO_LINE = ['L43', 'L43', 'L54', 'L54'][ZINDEX]
nu_co = sc.nu_co43 if CO_LINE == 'L43' else sc.nu_co54
K_BINS = np.logspace(-3, 3, 21)
MPC_TO_M = u.Mpc.to(u.m)
JY_CONVERSION = 1/u.Jy.to(u.W / u.m**2 / u.Hz)
EUCLID_FLUXCUT = 6.9e-17  # erg/s/cm^2
EUCLID_FLUXCUT_SCATTER = 2.8e-17  # scatter in flux cut
NIIcorr = 0.4  # [NII] contamination fraction for H-alpha
LINES = ['CII', 'HI', 'CO', 'Halpha', 'galaxy']
SFR_SCATTER = 0.3
SFR_AMP_SCATTER = 0.1
IMFFAC = 1 # / 0.63 # divide for Salpeter, 1 for Chabrier

SAVEFILENAME = f'power_spectra_ensemble_{Z_MODELING:.2f}.npz'
OUTDIR = f'../outs/new/{Z_MODELING:.2f}/'
FIGDIR = f'../outs/new/{Z_MODELING:.2f}/'
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
Path(FIGDIR).mkdir(parents=True, exist_ok=True)

def get_memory_usage():
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    return None


def log_memory(message=""):
    """Log current memory usage."""
    if PSUTIL_AVAILABLE:
        memory_mb = get_memory_usage()
        print(f"Memory usage {message}: {memory_mb:.1f} MB")
    else:
        print(f"Memory logging disabled {message}")


class PowerSpectrumAnalyzer:
    """Class to handle power spectrum calculations and analysis."""
    
    def __init__(self, sim_name='TNG100-1', random_seed=RANDOM_SEED, z=Z_MODELING):
        """Initialize the analyzer with simulation data."""
        self.sim_handler = sim.SimHandler(sim_name)
        self.rng = np.random.default_rng(random_seed)
        self.galfluxcut = self.rng.normal(EUCLID_FLUXCUT, EUCLID_FLUXCUT_SCATTER)
        self.pixel_size = self.sim_handler.box_edge_no_h / GRID_RESOLUTION
        self.z = z
        self.snap = self.sim_handler.get_snap_from_z(self.z)
        
        # Set up k-space bins
        self.bins = K_BINS
        self.bin_centers = (self.bins[1:] + self.bins[:-1]) / 2
        self.k = 2 * np.pi * self.bin_centers
        
        # Initialize cosmological parameters
        self._setup_cosmology()
        
        # Storage for grids and power spectra
        self.grids = {}
        self.power_spectra = {}
        self.cross_power_spectra = {}
        
        self.means_and_densities = {}
        self.props = ["sfr", "sfr_behroozi", "mass", "m_stars_wind", "MHI", "ngal"]
        self.lines = ["LCII", "LHI", CO_LINE, "LHa"]

    def _setup_cosmology(self):
        """Set up cosmological distance calculations."""
        self.d_comoving = self.snap.cosmo.comoving_distance(self.z).value
        self.d_luminosity = (1 + self.z) * self.d_comoving
        self.hubble_factor = 1000 * self.snap.cosmo.H(self.z).value
    
    def _calculate_conversion_factor(self, nu_rest):
        """Calculate conversion factor to Jy/sr for voxel formalism."""
        y_factor = sc.c / nu_rest * (1 + self.z)**2 / self.hubble_factor
        conversion = (sc.Lsun_to_W / (4 * np.pi * self.d_luminosity**2) * 
                     self.d_comoving**2 * y_factor / self.pixel_size**3 *
                     1 / MPC_TO_M**2 * JY_CONVERSION)
        return conversion
    
    def _calculate_Ldensity_to_Jy_conversion(self, nu_rest):
        """Conversion factor from Lsun/Mpc^3 to Jy/sr at redshift self.z."""
        # conversion factor
        conversion = (con.c.to(u.m/u.s).value / (4.0 * np.pi)) * \
            (sc.Lsun_to_W / MPC_TO_M**3) / (self.snap.cosmo.H(self.z).to(1/u.s).value * nu_rest) * JY_CONVERSION
        return conversion
    
    def _calculate_hi_muK_conversion(self):
        """Calculate conversion factor for HI from Jy/sr to muK."""
        nu_21_obs = sc.nu_hi / (1 + self.z)
        conv_muK = (1e6 * (sc.c**2 / (2 * con.k_B * nu_21_obs**2))
                .to(u.K/u.J).value / JY_CONVERSION)
        return conv_muK
    
    def setup_line_properties(self):
        """Set up all line emission properties."""
        # Star formation rate (Behroozi)
        self.snap.make_property(
            prop_behroozi_sfr, 
            other_kws={'rng': self.rng, 'sigma_scatter': SFR_SCATTER, 
                       'amp_scatter': SFR_AMP_SCATTER, 'imffac': IMFFAC}, 
            overwrite=True, 
            rename='sfr_behroozi'
        )
        
        # CII line emission
        self.snap.make_property(
            prop_delooze_cii, 
            other_kws={'rng': self.rng}, 
            overwrite=True, 
            rename='LCII', 
            kw_remap={'sfr': 'sfr_behroozi'}
        )

        # neutral hydrogen mass
        self.snap.make_property(
            prop_mass_hi, 
            other_kws={'rng': self.rng, 'cosmo': cosmo},
            overwrite=True
        )

        # HI line emission
        self.snap.make_property(
            prop_li_hi, 
            overwrite=True, 
        )
        
        # CO line emission
        self.snap.make_property(
            prop_co_sled, 
            other_kws={'rng': self.rng, 'fircor': 0.6}, 
            overwrite=True, 
            kw_remap={'sfr': 'sfr_behroozi'}
        )
        
        # H-alpha line emission
        self.snap.make_property(
            prop_halpha,
            other_kws={'rng': self.rng},
            overwrite=True,
            kw_remap={'sfr': 'sfr_behroozi'}
        )
        
        # Galaxy survey (e.g., Euclid-like)
        self.snap.make_property(
            prop_galaxy_survey,
            other_kws={'fluxcut': self.galfluxcut,
                    'NIIcorr': NIIcorr, 'cosmo': cosmo},
            overwrite=True,
        )
    
    def compute_mean_properties(self):

        volume = self.snap.box_edge ** 3 # Mpc^3

        self.means_and_densities['props'] = {}
        self.means_and_densities['lines'] = {}

        for prop in self.props:
            propvals = self.snap.return_property(prop)
            self.means_and_densities['props'][prop] = {
                "mean": np.nanmean(propvals),
                "median": np.nanmedian(propvals),
                "std": np.nanstd(propvals),
                "density": np.nansum(propvals) / volume
            }

        nulines = {"LCII": sc.nu_cii, 
                   "LHI": sc.nu_hi, 
                   CO_LINE: nu_co,
                   "LHa": sc.nu_halpha}

        for line in self.lines:
            conv_factor = self._calculate_Ldensity_to_Jy_conversion(nulines[line])
            linevals = self.snap.return_property(line)
            self.means_and_densities['lines'][line] = {
                "meanInu_Lsun/Mpc3": np.nansum(linevals) / volume,
                "meanInu_Jy/sr": np.nansum(linevals)/volume * conv_factor,
            }
        
        self.means_and_densities['galfluxcut'] = self.galfluxcut

    def create_line_grid(self, line_type):
        """Create grid for specific line emission type."""
        if line_type == 'CII':
            conv_factor = self._calculate_conversion_factor(sc.nu_cii)
            grid = self.snap.grid('LCII', res=self.pixel_size, norm=conv_factor)
        elif line_type == 'HI':
            conv_factor = self._calculate_conversion_factor(sc.nu_hi)
            conv_muK = self._calculate_hi_muK_conversion()
            grid = self.snap.grid('LHI', res=self.pixel_size, 
                                norm=conv_factor)# * conv_muK)
        elif line_type == 'CO':
            conv_factor = self._calculate_conversion_factor(nu_co)
            grid = self.snap.grid(CO_LINE, res=self.pixel_size, norm=conv_factor)
        elif line_type == 'Halpha':
            conv_factor = self._calculate_conversion_factor(sc.nu_halpha)
            grid = self.snap.grid('LHa', res=self.pixel_size, norm=conv_factor)
        elif line_type == 'galaxy':
            # Galaxy overdensity field
            grid = self.snap.grid('ngal', res=self.pixel_size)
            grid.grid /= np.mean(grid.grid)
            grid.grid -= 1.0
        else:
            raise ValueError(f"Unknown line type: {line_type}")
        
        self.grids[line_type] = grid
        return grid
    
    def calculate_power_spectrum(self, line_type):
        """Calculate 1D power spectrum for a line type."""
        if line_type not in self.grids:
            raise ValueError(f"Grid for {line_type} not created")
        
        grid = self.grids[line_type]
        ps = grid.power_spectrum(in_place=False, normalize=True)
        
        # Spherical average
        _, ps1d = ps.spherical_average(ax=[0, 1, 2], shells=self.bins)
        ps1d_normalized = ps1d[:, 0] / np.prod(grid.side_length)
        
        self.power_spectra[line_type] = ps1d_normalized
        
        # Clean up intermediate power spectrum object to save memory
        del ps
        gc.collect()
        
        return ps1d_normalized
    
    def calculate_cross_power_spectrum(self, line1, line2):
        """Calculate cross power spectrum between two line types."""
        if line1 not in self.grids or line2 not in self.grids:
            raise ValueError("Both grids must be created before cross-correlation")
        
        grid1 = self.grids[line1]
        grid2 = self.grids[line2]
        
        cross_ps = grid1.power_spectrum(cross_grid=grid2, in_place=False, 
                                    normalize=True)
        
        # Spherical average
        _, ps1d = cross_ps.spherical_average(ax=[0, 1, 2], shells=self.bins)
        ps1d_normalized = ps1d[:, 0] / np.prod(grid1.side_length)
        
        cross_key = f"{line1}x{line2}"
        self.cross_power_spectra[cross_key] = ps1d_normalized
        
        # Clean up intermediate cross power spectrum object
        del cross_ps
        gc.collect()
        
        return ps1d_normalized
    
    def run_full_analysis(self):
        """Run complete power spectrum analysis for all line types."""
        print("Setting up line properties...")
        self.setup_line_properties()
        
        print("computing mean properties")
        self.compute_mean_properties()

        print("Creating grids and calculating power spectra...")
        line_types = LINES
        
        # Create grids and calculate auto power spectra
        for line_type in line_types:
            self.create_line_grid(line_type)
            self.calculate_power_spectrum(line_type)
        
        # Calculate cross power spectra
        for line1, line2 in product(line_types, repeat=2):
            if line1 < line2:  # Avoid duplicate calculations
                self.calculate_cross_power_spectrum(line1, line2)

        # Clear grids to free memory - we only need the power spectra now
        self.grids.clear()
        gc.collect()
        
        print("Analysis complete!")
    
    def cleanup_memory(self):
        """Clean up large objects to free memory."""
        # Clear grids (largest memory users)
        self.grids.clear()
        
        # Clear simulation handler caches if they exist
        if hasattr(self.sim_handler, 'clear_cache'):
            self.sim_handler.clear_cache()
        
        # Force garbage collection
        gc.collect()
        print(f"Cleaned up analyzer memory for seed {self.rng.bit_generator._seed_seq.entropy}")


def run_single_analysis(seed):
    """Run a single analysis with given random seed - for multiprocessing."""
    log_memory(f"before analysis (seed {seed})")
    
    print(f"Running analysis with seed {seed}")
    analyzer = PowerSpectrumAnalyzer(random_seed=seed)
    analyzer.run_full_analysis()
    
    log_memory(f"after analysis (seed {seed})")
    
    # Force garbage collection after each analysis to free memory
    gc.collect()
    
    log_memory(f"after cleanup (seed {seed})")
    
    return analyzer


def load_existing_results(save_path=OUTDIR, filename=SAVEFILENAME):
    """Load existing power spectra results from a numpy file, if it exists.
    Returns a dict or None if file doesn't exist.
    """
    lines = LINES
    cross_keys = [f"{l1}x{l2}" for l1, l2 in product(lines, repeat=2) if l1 < l2]
    try:
        filepath = f"{save_path}{filename}"
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        existing_results = {
            'k': data['k'],
            'seeds': data['seeds'],
            'means_and_densities': data['means_and_densities'],
            'metadata': {
                'n_realizations': int(data['n_realizations']),
                'random_seed_base': int(data['random_seed_base']),
                'z_modeling': float(data['z_modeling']),
                'grid_resolution': int(data['grid_resolution']),
                'CO_line': data.get('CO_line', 'L43'),  # Default to CO(4-3) if not present
            }
        }
        # Load auto power spectra
        for line in lines:
            key = f'ps_{line.lower()}'
            existing_results[key] = data[key]
        # Load cross power spectra
        for cross_key in cross_keys:
            key = f'ps_cross_{cross_key.lower()}'
            existing_results[key] = data[key]
        
        print(f"Found existing results with {len(data['seeds'])} realizations")
        return existing_results
    except FileNotFoundError:
        print("No existing results found")
        return None


def create_analyzer_from_cache(seed, cached_data):
    """Create a PowerSpectrumAnalyzer object from cached data.
    
    Parameters
    ----------
    seed : int
        The random seed for this realization
    cached_data : dict
        Dictionary containing cached power spectra data
        
    Returns
    -------
    analyzer : PowerSpectrumAnalyzer
        Analyzer object with power spectra populated from cache
    """
    lines = LINES
    cross_keys = [f"{l1}x{l2}" for l1, l2 in product(lines, repeat=2) if l1 < l2]
    # Find the index of this seed in cached data
    seed_idx = np.where(cached_data['seeds'] == seed)[0][0]
    
    # Create analyzer (but don't run full analysis)
    analyzer = PowerSpectrumAnalyzer(random_seed=seed)
    
    # Populate power spectra from cached data
    analyzer.power_spectra = {line: cached_data[f'ps_{line.lower()}'][seed_idx] for line in lines}
    analyzer.cross_power_spectra = {line: cached_data[f'ps_cross_{line.lower()}'][seed_idx] for line in cross_keys}
    
    analyzer.means_and_densities = cached_data['means_and_densities'][seed_idx]
    
    print(f"Loaded cached results for seed {seed}")
    return analyzer


def save_power_spectra(analyzers, seeds, save_path=OUTDIR, filename=SAVEFILENAME):
    """Save all power spectra data to a numpy file.
    
    Parameters
    ----------
    analyzers : list
        List of PowerSpectrumAnalyzer objects
    seeds : list
        List of random seeds used for each analyzer
    save_path : str
        Directory to save the file
    filename : str
        Name of the output file
    """
    lines = LINES
    cross_keys = [f"{l1}x{l2}" for l1, l2 in product(lines, repeat=2) if l1 < l2]
    # Get k bins from first analyzer (should be same for all)
    k = analyzers[0].k
    
    # Initialize arrays to store all power spectra
    n_realizations = len(analyzers)
    n_k = len(k)
    
    # Auto power spectra
    all_auto_ps = {line: np.zeros((n_realizations, n_k)) for line in lines}
    all_cross_ps = {cross_key: np.zeros((n_realizations, n_k)) for cross_key in cross_keys}
    
    # Mean properties data - store as object array to preserve structure
    means_and_densities = np.empty(n_realizations, dtype=object)
    
    # Fill arrays with data from each analyzer
    for i, analyzer in enumerate(analyzers):
        for line in lines:
            all_auto_ps[line][i] = analyzer.power_spectra[line]
        for cross_key in cross_keys:
            all_cross_ps[cross_key][i] = analyzer.cross_power_spectra[cross_key]
        means_and_densities[i] = analyzer.means_and_densities
        
        means_and_densities[i] = analyzer.means_and_densities
    
    # Save to compressed numpy file
    output_file = f"{save_path}{filename}"
    
    output_dict = {
        'k': k,
        'seeds': np.array(seeds),
        'means_and_densities': means_and_densities,
        'n_realizations': n_realizations,
        'random_seed_base': RANDOM_SEED,
        'z_modeling': Z_MODELING,
        'grid_resolution': GRID_RESOLUTION,
        'CO_line': CO_LINE,
    }
    for line in lines:
        output_dict[f'ps_{line.lower()}'] = all_auto_ps[line]
    for cross_key in cross_keys:
        output_dict[f'ps_cross_{cross_key.lower()}'] = all_cross_ps[cross_key]
    
    with open(output_file, 'wb') as f:
        pickle.dump(output_dict, f)
    
    print(f"Power spectra data saved to: {output_file}")
    print(f"Contains {n_realizations} realizations with {n_k} k-bins each")


def plot_power_spectra(analyzers, fig_path=FIGDIR):
    """Plot power spectra and cross-correlation results."""
    analyzers = analyzers if isinstance(analyzers, list) else [analyzers]
    lines = LINES
    cross_keys = [f"{l1}x{l2}" for l1, l2 in product(lines, repeat=2) if l1 < l2]
    colors = [f"C{i}" for i in range(len(lines) + len(cross_keys))]
    
    # Main power spectrum plot
    fig0, ax0 = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    fig0.subplots_adjust(wspace=0.4)
    
    # cross correlation coefficient plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 6))
    
    # fill between plot
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    
    # cross correlation coefficient with uncertainty bands
    fig4, ax4 = plt.subplots(1, 1, figsize=(7, 6))
    
    for ax, fig in zip([ax0, ax2], [fig0, fig2]):
        # Left panel - P(k)
        ax[0].set(xlabel='k [Mpc$^{-1}$]',
                ylabel='P(k) [$Jy^2/sr^2/Mpc^3$]', #or muK$^2$/Mpc$^3$ or Jy/sr muK/Mpc$^3$]',
                xscale='log', yscale='log', ylim=(1e1, 1e12))
        ax[0].grid()
        
        # Right panel - k^3 P(k) / 2π²
        ax[1].set(xlabel='k [Mpc$^{-1}$]',
                ylabel='k$^3$P(k)/2π$^2$ [$Jy^2/sr^2$]', # or muK$^2$ or Jy/sr muK]',
                xscale='log', yscale='log', ylim=(1e0, 1e13))
        ax[1].grid()
    
    for ax, fig in zip([ax1, ax4], [fig1, fig4]):
        ax.set_xscale('log')
        ax.grid()
        ax.set_ylim(.2, 1.0)
        ax.set_xlabel('k [Mpc$^{-1}$]')
        ax.set_ylabel('Cross correlation coefficient')
        ax.set_title('Cross correlation coefficient $P_{i \\times j} / \\sqrt{P_{i}P_{j}}$' f' at z={Z_MODELING:.2f}')
    
    for i, analyzer in enumerate(analyzers):
        k = analyzer.k
        # plot auto and cross power spectra
        plot_data = []
        for i, line in enumerate(lines):
            if line not in analyzer.power_spectra:
                raise ValueError(f"Power spectrum for {line} not found in analyzer")
            plot_data.append((analyzer.power_spectra[line], colors[i], f"$P_{{{line}}}$"))
        
        for j, cross_key in enumerate(cross_keys):
            if cross_key not in analyzer.cross_power_spectra:
                raise ValueError(f"Cross power spectrum for {cross_key} not found in analyzer")
            plot_data.append((analyzer.cross_power_spectra[cross_key], colors[len(lines)+j], f"$P_{{{cross_key}}}$"))
        
        for ps_data, color, label in plot_data:
            lab = label if i == 0 else None  # Only add label for the first plot
            ax0[0].plot(k, ps_data, lw=1, color=color, label=lab, alpha=0.25)
            ax0[1].plot(k, k**3 / (2 * np.pi**2) * ps_data, lw=1, color=color, alpha=0.25, label=lab)
        
        ax0[0].legend(loc='upper right')
        ax0[1].legend(loc='upper right')
        
        for j, cross_key in enumerate(cross_keys):
            if "galaxy" in cross_key:
                line1, line2 = cross_key.split('x')[0], "galaxy"
            else:
                line1, line2 = cross_key.split('x')
            ps_cross = analyzer.cross_power_spectra[cross_key]
            ps1 = analyzer.power_spectra[line1]
            ps2 = analyzer.power_spectra[line2]
            prod_ps = np.sqrt(ps1 * ps2)
            lab = f"$i$ = {line1}, $j$ = {line2}" if i == 0 else None
            ax1.plot(k, ps_cross / prod_ps, lw=1, label=lab, alpha=0.25, color=colors[len(lines)+j])
        ax1.legend(loc='upper right') if i == 0 else None

    fig0.suptitle(f'Power Spectra for CII, HI, and CO at z={Z_MODELING} (N={len(analyzers)})')
    fig0.savefig(f'{fig_path}cross_ps_{Z_MODELING:.2f}.png', dpi=300, bbox_inches='tight')
    
    # Calculate statistics for fill_between plot
    n_realizations = len(analyzers)
    if n_realizations > 1:
        k = analyzers[0].k
        
        ps_data_sets = []
        
        for i, line in enumerate(lines):
            all_ps = np.array([analyzer.power_spectra[line] for analyzer in analyzers])
            ps_data_sets.append((all_ps, colors[i], f"$P_{{{line}}}$"))
        
        for j, cross_key in enumerate(cross_keys):
            all_ps = np.array([analyzer.cross_power_spectra[cross_key] for analyzer in analyzers])
            ps_data_sets.append((all_ps, colors[len(lines)+j], f"$P_{{{cross_key}}}$"))
    
        # Calculate percentiles (16th, 50th, 84th for 1-sigma)
        percentiles = [16, 50, 84]
        
        for ps_data, color, label in ps_data_sets:
            # Calculate percentiles
            p16, p50, p84 = np.percentile(ps_data, percentiles, axis=0)
            
            # Plot median and fill between 16th-84th percentiles
            ax2[0].plot(k, p50, lw=2, color=color, label=label)
            ax2[0].fill_between(k, p16, p84, color=color, alpha=0.2)
            
            # Same for dimensionless power spectrum
            p16_dim = k**3 / (2 * np.pi**2) * p16
            p50_dim = k**3 / (2 * np.pi**2) * p50
            p84_dim = k**3 / (2 * np.pi**2) * p84
            
            ax2[1].plot(k, p50_dim, lw=2, color=color, label=label)
            ax2[1].fill_between(k, p16_dim, p84_dim, color=color, alpha=0.2)
        
        ax2[0].legend(loc='upper right')
        ax2[1].legend(loc='upper right')
        
        fig2.suptitle(f'Power Spectra with percentile bands at z={Z_MODELING} (N={n_realizations})')
        fig2.savefig(f'{fig_path}cross_ps_fb_{Z_MODELING:.2f}.png', dpi=300, bbox_inches='tight')
        
        for j, cross_key in enumerate(cross_keys):
            if "galaxy" in cross_key:
                line1, line2 = cross_key.split('x')[0], "galaxy"
            else:
                line1, line2 = cross_key.split('x')
            all_ps_cross = np.array([analyzer.cross_power_spectra[cross_key] for analyzer in analyzers])
            all_ps1 = np.array([analyzer.power_spectra[line1] for analyzer in analyzers])
            all_ps2 = np.array([analyzer.power_spectra[line2] for analyzer in analyzers])
            
            prod_ps = np.sqrt(all_ps1 * all_ps2)
            
            # Calculate percentiles for cross-correlation coefficients
            coeffs = all_ps_cross / prod_ps
            p16, p50, p84 = np.percentile(coeffs, percentiles, axis=0)
            
            lab = f"$i$ = {line1}, $j$ = {line2}"
            ax4.plot(k, p50, lw=2, label=lab, color=colors[len(lines)+j])
            ax4.fill_between(k, p16, p84, alpha=0.2, color=colors[len(lines)+j])

        ax4.legend(loc='upper right')

        fig4.savefig(f'{fig_path}cross_coeff_fb_{Z_MODELING:.2f}.png', dpi=300, bbox_inches='tight')
        
        gc.collect()
    else:
        print("Need more than 1 realization for uncertainty bands")
    
    # Configure and show the cross-correlation coefficient plot
    fig1.savefig(f'{fig_path}cross_coeff_{Z_MODELING:.2f}.png', dpi=300, bbox_inches='tight')
    
    # Final cleanup
    gc.collect()
    print("All plots saved and memory cleaned up")
    
    # plt.show()
    plt.close('all')

def get_optimal_process_count(seeds_to_compute, memory_per_process_gb=6, system_memory_gb=8):
    """
    Determine optimal number of processes based on available memory.
    
    Parameters
    ----------
    seeds_to_compute : list
        List of seeds to compute
    memory_per_process_mb : float
        Estimated memory per process in MB
    
    Returns
    -------
    int
        Optimal number of processes
    """
    n_cpu = os.cpu_count()
    n_seeds = len(seeds_to_compute)
    memory_per_process_mb = memory_per_process_gb * 1024  # Convert GB to MB
    
    if PSUTIL_AVAILABLE:
        # Get available memory (leave X GB for system)
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024 - system_memory_gb * 1024
        memory_limited_processes = max(1, int(available_memory_mb / memory_per_process_mb))
        
        # Choose minimum of CPU-limited, memory-limited, and seeds-limited
        optimal = min(n_cpu // 2, memory_limited_processes, n_seeds)
        
        print(f"Available memory: {available_memory_mb:.0f} MB")
        print(f"Estimated memory per process: {memory_per_process_mb} MB")
        print(f"CPU-limited processes: {n_cpu // 2}")
        print(f"Memory-limited processes: {memory_limited_processes}")
        print(f"Optimal processes: {optimal}")
        
        return optimal
    else:
        # Fallback to CPU-based calculation
        return min(n_seeds, n_cpu // 2)


def main(N=3, cache_on=True, seeds=None):
    """Main execution function."""
    # Generate seeds for each realization
    if seeds is None:
        seeds = [RANDOM_SEED + i for i in range(N)]
    
    # Check for existing results if caching is enabled
    analyzers = []
    seeds_to_compute = []
    
    if cache_on:
        print("Checking for existing cached results...")
        existing_data = load_existing_results()
        
        if existing_data is not None:
            existing_seeds = set(existing_data['seeds'])
            
            for seed in seeds:
                if seed in existing_seeds:
                    # Load from cache
                    analyzer = create_analyzer_from_cache(seed, existing_data)
                    analyzers.append(analyzer)
                else:
                    # Need to compute
                    seeds_to_compute.append(seed)
            
            print(f"Found {len(analyzers)} cached results, need to compute {len(seeds_to_compute)} new ones")
        else:
            seeds_to_compute = seeds
    else:
        seeds_to_compute = seeds
    
    # Compute any remaining seeds
    if seeds_to_compute:
        # Determine optimal number of processes based on memory and CPU
        n_processes = get_optimal_process_count(seeds_to_compute)
        print(f"Running {len(seeds_to_compute)} realizations using {n_processes} processes...")
        
        log_memory("before multiprocessing")
        
        # Run analyses in parallel
        with Pool(processes=n_processes) as pool:
            new_analyzers = pool.map(run_single_analysis, seeds_to_compute)
        
        analyzers.extend(new_analyzers)
        print(f"Computed {len(seeds_to_compute)} new realizations!")
        
        # Force garbage collection after multiprocessing
        gc.collect()
        log_memory("after multiprocessing")
    
    # Sort analyzers by seed to maintain consistent ordering
    analyzers.sort(key=lambda x: x.rng.bit_generator._seed_seq.entropy)
    
    print(f"Total of {len(analyzers)} realizations ready!")
    
    # Save power spectra data (this will update the cache with any new results)
    if seeds_to_compute:  # Only save if we computed new results
        print("Saving power spectra data...")
        save_power_spectra(analyzers, seeds)
    
    print(f"Plotting results from {len(analyzers)} realizations...")
    
    log_memory("before plotting")
    
    # Create plots with all realizations
    plot_power_spectra(analyzers)
    
    log_memory("after plotting")


if __name__ == "__main__":
    N = 3
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            print("Invalid argument, using default N")
    main(N=N, cache_on=True)
    print("All complete!")
