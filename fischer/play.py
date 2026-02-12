import numpy as np
import emcee
from getdist import plots, MCSamples

# -----------------------------
# Constants for all three likelihoods
Pd = 1.0
P1, P2 = 1.0, 1.0
sigma1, sigma2 = 1, 1 / 30
Px = 1.0
sigmax = 1 / 10

# -----------------------------
# Priors (Gaussian, peaky at 1)
prior_mean = np.array([1.0, 1.0])
prior_sigma = np.array([0.5, 0.5])  #  narrow = peaky

def log_uniform(theta):
    theta = np.array(theta)
    if np.any(theta <= 0) or np.any(theta >= 2):
        return -np.inf
    return 0.0

def log_gauss(theta):
    theta = np.array(theta)
    if np.any(theta <= 0):
        return -np.inf
    return -0.5 * np.sum(((theta - prior_mean)/prior_sigma)**2)

# -----------------------------
# Likelihoods
def loglike_L1(theta):
    bI1, bI2 = theta
    model1 = bI1**2 * Pd
    model2 = bI2**2 * Pd
    chi2 = ((model1 - P1)**2 / sigma1**2 +
            (model2 - P2)**2 / sigma2**2)
    return -0.5 * chi2

def loglike_L2(theta):
    bI1, bI2 = theta
    model1 = bI1**2 * Pd
    model2 = bI2**2 * Pd
    modelx = bI1 * bI2 * Pd
    chi2 = ((model1 - P1)**2 / sigma1**2 +
            (model2 - P2)**2 / sigma2**2 +
            (modelx - Px)**2 / sigmax**2)
    return -0.5 * chi2

def loglike_L3(theta):
    bI1, bI2 = theta
    modelx = bI1 * bI2 * Pd
    chi2 = ((modelx - Px)**2 / sigmax**2)
    return -0.5 * chi2

# -----------------------------
# Log-posterior wrapper
def make_logpost(loglike):
    def log_post(theta):
        lp = log_gauss(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + loglike(theta)
    return log_post

# -----------------------------
# Run MCMC for a given log-posterior
def run_mcmc(log_post, ndim=2, nwalkers=100, nsteps=10000, discard=1000):
    pos = prior_mean + 0.01*np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post)
    sampler.run_mcmc(pos, nsteps, progress=True)
    samples = sampler.get_chain(discard=discard, flat=True)
    return samples

# -----------------------------
# Sample all three likelihoods
samples_L1 = run_mcmc(make_logpost(loglike_L1))
samples_L2 = run_mcmc(make_logpost(loglike_L2))
samples_L3 = run_mcmc(make_logpost(loglike_L3))

# -----------------------------
# Convert to GetDist format
names = ["bI1", "bI2"]
labels = [" . $ bias * mean intensity (low SNR tracer) $ . ", \
    " . $ bias * mean intensity (high SNR tracer) $ . "]
gd_L1 = MCSamples(samples=samples_L1, names=names, labels=labels)
gd_L2 = MCSamples(samples=samples_L2, names=names, labels=labels)
gd_L3 = MCSamples(samples=samples_L3, names=names, labels=labels)

# -----------------------------
# Make single figure with all three contours
g = plots.get_subplot_plotter(width_inch=6)
g.plot_2d([gd_L1, gd_L2, gd_L3], ['bI1', 'bI2'], filled=[False, True, False],
        labels= ["L1", "L2", "L3"], colors=['blue', 'green', 'red'], 
        lims=[0.5, 1.5, 0.5, 1.5])
g.export("three_contours.png")
