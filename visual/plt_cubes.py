import numpy as np
from matplotlib import pyplot as plt

grids = np.load("./tng100_z1p173_grids.npz")
keys = list(grids.keys())

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
axes = axes.flatten()

cmaps = ['Greens', 'Blues', 'Purples', 'Greys']

for i, k in enumerate(keys):
    print("Processing ", k)
    data = grids[k][:, :, :, 0]
    
    cenx, ceny = 250, 250
    size = 50
    data = data[cenx-size:cenx+size, ceny-size:ceny+size]
    data = data.sum(axis=2)
    
    if i == 3:
        data += (-1 * data.min() + 1e-6)
        data /= data.max()
    
    data = np.log10(data + 1e-6)
    vmin, vmax = np.percentile(data, [5, 95]) if i != 3 else (None, None)
    axes[i].imshow(data.T, origin='lower', cmap=cmaps[i], vmin=vmin, vmax=vmax)
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])
    axes[i].grid(True)
    
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig("./plt_tng100_z1p173_cubes.png", dpi=300)
plt.close()