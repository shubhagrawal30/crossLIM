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
import os, sys, gc
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
    prop_li_hi, prop_co_sled, prop_mass_hi
)


# Configuration constants
ZINDEX = 0
RANDOM_SEED = 1511
GRID_RESOLUTION = 450  # pixels per box edge
Z_MODELING = [0.644, 0.888, 1.173, 1.499][ZINDEX]
CO_LINE = ['L43', 'L43', 'L54', 'L54'][ZINDEX]
nu_co = sc.nu_co43 if CO_LINE == 'L43' else sc.nu_co54
K_BINS = np.logspace(-3, 3, 21)
MPC_TO_M = 3.0857e22
JY_CONVERSION = 1e26

SAVEFILENAME = f'power_spectra_ensemble_{Z_MODELING:.2f}.npz'
OUTDIR = f'../outs/{Z_MODELING:.2f}/'
FIGDIR = f'../figs/{Z_MODELING:.2f}/'
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
        self.props = ["sfr", "sfr_behroozi", "mass", "m_stars_wind", "MHI"]
        self.lines = ["LCII", "LHI", CO_LINE]

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
            other_kws={'rng': self.rng}, 
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

        nulines = {"LCII": sc.nu_cii, "LHI": sc.nu_hi, CO_LINE: nu_co}

        for line in self.lines:
            conv_factor = self._calculate_Ldensity_to_Jy_conversion(nulines[line])
            linevals = self.snap.return_property(line)
            self.means_and_densities['lines'][line] = {
                "meanInu_Lsun/Mpc3": np.nansum(linevals) / volume,
                "meanInu_Jy/sr": np.nanmean(linevals)/volume * conv_factor,
            }

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
        line_types = ['CII', 'HI', 'CO']
        
        # Create grids and calculate auto power spectra
        for line_type in line_types:
            self.create_line_grid(line_type)
            self.calculate_power_spectrum(line_type)
        
        # Calculate cross power spectra
        self.calculate_cross_power_spectrum('CII', 'HI')
        self.calculate_cross_power_spectrum('CII', 'CO')
        self.calculate_cross_power_spectrum('CO', 'HI')

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
    try:
        filepath = f"{save_path}{filename}"
        data = np.load(filepath, allow_pickle=True)
        existing_results = {
            'k': data['k'],
            'seeds': data['seeds'],
            'ps_cii': data['ps_cii'],
            'ps_hi': data['ps_hi'],
            'ps_co': data['ps_co'],
            'ps_cross_hi': data['ps_cross_hi'],
            'ps_cross_co': data['ps_cross_co'],
            'ps_cross_cohi': data['ps_cross_cohi'],
            'means_and_densities': data['means_and_densities'],
            'metadata': {
                'n_realizations': int(data['n_realizations']),
                'random_seed_base': int(data['random_seed_base']),
                'z_modeling': float(data['z_modeling']),
                'grid_resolution': int(data['grid_resolution']),
                'CO_line': data.get('CO_line', 'L43'),  # Default to CO(4-3) if not present
            }
        }
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
    # Find the index of this seed in cached data
    seed_idx = np.where(cached_data['seeds'] == seed)[0][0]
    
    # Create analyzer (but don't run full analysis)
    analyzer = PowerSpectrumAnalyzer(random_seed=seed)
    
    # Populate power spectra from cached data
    analyzer.power_spectra = {
        'CII': cached_data['ps_cii'][seed_idx],
        'HI': cached_data['ps_hi'][seed_idx],
        'CO': cached_data['ps_co'][seed_idx]
    }
    
    analyzer.cross_power_spectra = {
        'CIIxHI': cached_data['ps_cross_hi'][seed_idx],
        'CIIxCO': cached_data['ps_cross_co'][seed_idx],
        'COxHI': cached_data['ps_cross_cohi'][seed_idx]
    }
    
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
    # Get k bins from first analyzer (should be same for all)
    k = analyzers[0].k
    
    # Initialize arrays to store all power spectra
    n_realizations = len(analyzers)
    n_k = len(k)
    
    # Auto power spectra
    ps_cii = np.zeros((n_realizations, n_k))
    ps_hi = np.zeros((n_realizations, n_k))
    ps_co = np.zeros((n_realizations, n_k))
    
    # Cross power spectra
    ps_cross_hi = np.zeros((n_realizations, n_k))
    ps_cross_co = np.zeros((n_realizations, n_k))
    ps_cross_cohi = np.zeros((n_realizations, n_k))
    
    # Mean properties data - store as object array to preserve structure
    means_and_densities = np.empty(n_realizations, dtype=object)
    
    # Fill arrays with data from each analyzer
    for i, analyzer in enumerate(analyzers):
        ps_cii[i] = analyzer.power_spectra['CII']
        ps_hi[i] = analyzer.power_spectra['HI']
        ps_co[i] = analyzer.power_spectra['CO']
        ps_cross_hi[i] = analyzer.cross_power_spectra['CIIxHI']
        ps_cross_co[i] = analyzer.cross_power_spectra['CIIxCO']
        ps_cross_cohi[i] = analyzer.cross_power_spectra['COxHI']
        
        means_and_densities[i] = analyzer.means_and_densities
    
    # Save to compressed numpy file
    output_file = f"{save_path}{filename}"
    np.savez_compressed(
        output_file,
        k=k,
        seeds=np.array(seeds),
        ps_cii=ps_cii,
        ps_hi=ps_hi,
        ps_co=ps_co,
        ps_cross_hi=ps_cross_hi,
        ps_cross_co=ps_cross_co,
        ps_cross_cohi=ps_cross_cohi,
        means_and_densities=means_and_densities,
        # Metadata
        n_realizations=n_realizations,
        random_seed_base=RANDOM_SEED,
        z_modeling=Z_MODELING,
        grid_resolution=GRID_RESOLUTION,
        CO_line=CO_LINE
    )
    
    print(f"Power spectra data saved to: {output_file}")
    print(f"Contains {n_realizations} realizations with {n_k} k-bins each")


def plot_power_spectra(analyzers, fig_path=FIGDIR):
    """Plot power spectra and cross-correlation results."""
    analyzers = analyzers if isinstance(analyzers, list) else [analyzers]
    
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
                xscale='log', yscale='log')
        ax[0].grid()
        
        # Right panel - k^3 P(k) / 2π²
        ax[1].set(xlabel='k [Mpc$^{-1}$]',
                ylabel='k$^3$P(k)/2π$^2$ [$Jy^2/sr^2$]', # or muK$^2$ or Jy/sr muK]',
                xscale='log', yscale='log')
        ax[1].grid()
    
    for ax, fig in zip([ax1, ax4], [fig1, fig4]):
        ax.set_xscale('log')
        ax.grid()
        ax.set_xlabel('k [Mpc$^{-1}$]')
        ax.set_ylabel('Cross correlation coefficient')
        ax.set_title('Cross correlation coefficient $P_{i \\times j} / \\sqrt{P_{i}P_{j}}$')
    
    for i, analyzer in enumerate(analyzers):
        k = analyzer.k
        ps_cii = analyzer.power_spectra['CII']
        ps_hi = analyzer.power_spectra['HI'] 
        ps_co = analyzer.power_spectra['CO']
        ps_cross_hi = analyzer.cross_power_spectra['CIIxHI']
        ps_cross_co = analyzer.cross_power_spectra['CIIxCO']
        ps_cross_cohi = analyzer.cross_power_spectra['COxHI']
        
        # Plot data
        plot_data = [
            (ps_cross_hi, 'k', '$P_{CII \\times HI}$'),
            (ps_cii, 'b', '$P_{CII}$'),
            (ps_hi, 'r', '$P_{HI}$'),
            (ps_co, 'g', '$P_{CO}$'),
            (ps_cross_co, 'm', '$P_{CII \\times CO}$'),
            (ps_cross_cohi, 'c', '$P_{CO \\times HI}$')
        ]
        
        for ps_data, color, label in plot_data:
            lab = label if i == 0 else None  # Only add label for the first plot
            ax0[0].plot(k, ps_data, lw=1, color=color, label=lab, alpha=0.25)
            ax0[1].plot(k, k**3 / (2 * np.pi**2) * ps_data, lw=1, color=color, alpha=0.25, label=lab)
        
        ax0[0].legend(loc='upper right')
        ax0[1].legend(loc='upper right')
        
        # Cross-correlation coefficient plot
        prod_ps_hi = np.sqrt(ps_cii * ps_hi)
        prod_ps_co = np.sqrt(ps_cii * ps_co)
        prod_ps_cohi = np.sqrt(ps_co * ps_hi)
        lab1 = r'$i$ = CII, $j$ = HI'
        lab2 = r'$i$ = CII, $j$ = CO'
        lab3 = r'$i$ = CO, $j$ = HI'
        ax1.plot(k, ps_cross_hi / prod_ps_hi, lw=1, color='k', label=lab1, alpha=0.25)
        ax1.plot(k, ps_cross_co / prod_ps_co, lw=1, color='m', label=lab2, alpha=0.25)
        ax1.plot(k, ps_cross_cohi / prod_ps_cohi, lw=1, color='c', label=lab3, alpha=0.25) 
        ax1.legend(loc='upper right') if i == 0 else None

    fig0.suptitle(f'Power Spectra for CII, HI, and CO at z={Z_MODELING} (N={len(analyzers)})')
    fig0.savefig(f'{fig_path}cross_ps.png', dpi=300, bbox_inches='tight')
    
    # Calculate statistics for fill_between plot
    n_realizations = len(analyzers)
    if n_realizations > 1:
        k = analyzers[0].k
        
        # Collect all power spectra
        all_ps_cii = np.array([analyzer.power_spectra['CII'] for analyzer in analyzers])
        all_ps_hi = np.array([analyzer.power_spectra['HI'] for analyzer in analyzers])
        all_ps_co = np.array([analyzer.power_spectra['CO'] for analyzer in analyzers])
        all_ps_cross_hi = np.array([analyzer.cross_power_spectra['CIIxHI'] for analyzer in analyzers])
        all_ps_cross_co = np.array([analyzer.cross_power_spectra['CIIxCO'] for analyzer in analyzers])
        all_ps_cross_cohi = np.array([analyzer.cross_power_spectra['COxHI'] for analyzer in analyzers])
        
        # Calculate percentiles (16th, 50th, 84th for 1-sigma)
        percentiles = [16, 50, 84]
        
        ps_data_sets = [
            (all_ps_cross_hi, 'k', '$P_{CII \\times HI}$'),
            (all_ps_cii, 'b', '$P_{CII}$'),
            (all_ps_hi, 'r', '$P_{HI}$'),
            (all_ps_co, 'g', '$P_{CO}$'),
            (all_ps_cross_co, 'm', '$P_{CII \\times CO}$'),
            (all_ps_cross_cohi, 'c', '$P_{CO \\times HI}$')
        ]
        
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
        fig2.savefig(f'{fig_path}cross_ps_fb.png', dpi=300, bbox_inches='tight')
        
        # Cross-correlation coefficient uncertainty bands
        all_cross_coeff_hi = all_ps_cross_hi / np.sqrt(all_ps_cii * all_ps_hi)
        all_cross_coeff_co = all_ps_cross_co / np.sqrt(all_ps_cii * all_ps_co)
        all_cross_coeff_cohi = all_ps_cross_cohi / np.sqrt(all_ps_co * all_ps_hi)
        
        # Calculate percentiles for cross-correlation coefficients
        coeff_hi_p16, coeff_hi_p50, coeff_hi_p84 = np.percentile(all_cross_coeff_hi, percentiles, axis=0)
        coeff_co_p16, coeff_co_p50, coeff_co_p84 = np.percentile(all_cross_coeff_co, percentiles, axis=0)
        coeff_cohi_p16, coeff_cohi_p50, coeff_cohi_p84 = np.percentile(all_cross_coeff_cohi, percentiles, axis=0)
        
        # Plot cross-correlation coefficients with uncertainty bands
        ax4.plot(k, coeff_hi_p50, lw=2, color='k', label=lab1)
        ax4.fill_between(k, coeff_hi_p16, coeff_hi_p84, color='k', alpha=0.2)

        ax4.plot(k, coeff_co_p50, lw=2, color='m', label=lab2)
        ax4.fill_between(k, coeff_co_p16, coeff_co_p84, color='m', alpha=0.2)

        ax4.plot(k, coeff_cohi_p50, lw=2, color='c', label=lab3)
        ax4.fill_between(k, coeff_cohi_p16, coeff_cohi_p84, color='c', alpha=0.2)

        ax4.legend(loc='upper right')

        fig4.savefig(f'{fig_path}cross_coeff_fb.png', dpi=300, bbox_inches='tight')
        
        # Clean up large arrays
        del all_ps_cii, all_ps_hi, all_ps_co, all_ps_cross_hi, all_ps_cross_co
        del all_cross_coeff_hi, all_cross_coeff_co
        gc.collect()
    else:
        print("Need more than 1 realization for uncertainty bands")
    
    # Configure and show the cross-correlation coefficient plot
    fig1.savefig(f'{fig_path}cross_coeff.png', dpi=300, bbox_inches='tight')
    
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
