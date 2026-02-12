import yt
import numpy as np
from matplotlib import pyplot as plt

grids = np.load("./tng100_z1p173_grids.npz")

keys = list(grids.keys())

for i, k in enumerate(keys):
    # if i != 3:
    #     continue
    print("Processing ", k)
    data = grids[k][:, :, :, 0]
    
    # downsample by factor of N, adding up N^3 cells
    # N = 2
    # data = data.reshape((data.shape[0]//N, N, data.shape[1]//N, N, data.shape[2]//N, N)).sum(axis=(1,3,5))
    
    # zoom into a smaller region
    cenx, ceny, cenz = 64, 128, 128
    size = 45
    data = data[cenx-size:cenx+size, ceny-size:ceny+size, cenz-size:cenz+size]
    
    if i == 3:
        data += (-1 * data.min() + 1e-6)
        data /= data.max()
    # plt.figure()
    # plt.hist(data.flatten(), bins=1000, log=True)
    # plt.xlabel(k)
    # plt.ylabel("Counts")
    # plt.show()
    
    
    centerx, centery, centerz = 0, 0, 0
    sidex, sidey, sidez = data.shape
    bbox = np.array([[centerx-sidex, centerx+sidex], [centery-sidey, centery+sidey], \
                    [centerz - sidez, centerz + sidez]])
    ds = yt.load_uniform_grid({"density": data}, data.shape, length_unit=None, bbox=bbox, nprocs=128)
    sc = yt.create_scene(ds, field='density')
    source = sc[0]
    if i != 3:
        source.tfh.set_log(True)
    print("Data max, min: ", data.max(), data.min())
    # source.tfh.set_bounds((.2, data.max()))
    source.tfh.grey_opacity = True
    source.tfh.alpha = .3
    cam_settings = {
        # "resolution": (4096, 4096),
        # "resolution": (1024, 1024),
        # "resolution": (128, 128),
        "zoom": 0.95,
        "position": (0.6, 1.0, 0.7),
        "roll": 0.0
    }
    cam = sc.add_camera(lens_type="plane-parallel")
    # cam.resolution = cam_settings["resolution"]
    cam.zoom(cam_settings["zoom"])
    cam.set_position(cam_settings["position"])
    cam.roll(cam_settings["roll"])
    cam.rotate(125 * np.pi / 180., rot_vector=[1, 0, 0])
    cam.rotate(5 * np.pi / 180., rot_vector=[0, 1, 0])
    
    sc.save("./tng100_z1p173_grids_%s.png" % k, sigma_clip=10.0)