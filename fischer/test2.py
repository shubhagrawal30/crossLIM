import numpy as np
from getdist import plots, MCSamples

# Constants
Pd = 1.0
P1, P2 = 1.0, 1.0
sigma1, sigma2 = 0.1, 0.1
Px = 1.0
sigmax = 0.1

# Define likelihood
def log_likelihood(theta):
    bI1, bI2 = theta
    model1 = bI1**2 * Pd
    model2 = bI2**2 * Pd
    modelx = bI1 * bI2 * Pd
    chi2 = ((model1 - P1)**2 / sigma1**2 +
            (model2 - P2)**2 / sigma2**2 +
            (modelx - Px)**2 / sigmax**2)
    return -0.5 * chi2

# Peaky at 1 priors
def log_prior(theta):
    bI1, bI2 = theta
    return 0.0 if 1/2 < bI1 < 2 and 1/2 < bI2 < 2 else -np.inf
prior_mean = np.array([1, 1])
prior_sigma = np.array([0.5, 0.5])
# Gaussian log-prior
def log_gauss(theta):
    theta = np.array(theta)
    # positivity constraint
    if np.any(theta <= 0):
        return -np.inf
    # Gaussian prior
    lp = -0.5 * np.sum(((theta - prior_mean) / prior_sigma)**2)
    return lp

def log_posterior(theta):
    lp = log_gauss(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# Run MCMC
import emcee
ndim, nwalkers = 2, 100
pos = np.array([1, 1]) + 0.01 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(pos, 50000, progress=True)

samples = sampler.get_chain(discard=5000, flat=True)

# Pass to GetDist
names = ["bI1", "bI2"]
labels = ["bI1", "bI2"]

gd_samples = MCSamples(samples=samples, names=names, labels=labels)

# Make triangle plot
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, filled=True)
g.export("triangle_plot2.png")

