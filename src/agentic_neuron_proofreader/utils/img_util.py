"""
Created on Thu May 21 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for reading and processing images.

"""

import json
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts

from agentic_neuron_proofreader.utils import util


class TensorStoreImage:
    """
    Class that reads images with the TensorStore library.
    """

    def __init__(self, img_path):
        """
        Instantiates a TensorStoreImage object.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        # Load image
        bucket_name, inner_path = util.parse_cloud_path(img_path)
        self.img = ts.open(
            {
                "driver": get_driver(img_path),
                "kvstore": {
                    "driver": get_storage_driver(img_path),
                    "bucket": bucket_name,
                    "path": inner_path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": 1000000000},
                    "cache_pool#remote": {"total_bytes_limit": 1000000000},
                    "data_copy_concurrency": {"limit": 8},
                },
                "recheck_cached_data": "open",
            }
        ).result()

        # Check for Google segmentation
        if "from_google" in img_path:
            self.img = self.img[ts.d[:].transpose[3, 2, 1, 0]]

        # Check dimensions
        while self.img.ndim < 5:
            self.img = self.img[ts.newaxis, ...]

    def read(self, voxel, shape):
        """
        Reads a patch from an image given a voxel coordinate and patch shape.

        Parameters
        ----------
        voxel : Tuple[int]
            Center of image patch to be read.
        shape : Tuple[int]
            Shape of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        return self.img[(0, 0, *get_slices(voxel, shape))].read().result()

    def shape(self):
        """
        Gets the shape of image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


# --- Helpers ---
def get_driver(img_path):
    """
    Gets the driver needed to read the image.

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if ".zarr" in img_path:
        return "zarr"
    elif is_precomputed(img_path):
        return "neuroglancer_precomputed"
    raise Exception(f"Invalid image path at {img_path}")


def get_slices(center, shape):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    center : Tuple[int]
        Center of image patch to be read.
    shape : Tuple[int]
        Shape of image patch to be read.

    Return
    ------
    Tuple[Slice[int]]
        Slice objects used to index into the image.
    """
    start = [int(c - d // 2) for c, d in zip(center, shape)]
    return tuple(slice(s, s + d) for s, d in zip(start, shape))


def get_storage_driver(img_path):
    """
    Gets the storage driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Image path to be checked.

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if util.is_s3_path(img_path):
        return "s3"
    elif util.is_gcs_path(img_path):
        return "gcs"
    else:
        raise ValueError(f"Unsupported path type: {img_path}")


def is_precomputed(img_path):
    """
    Checks if the path points to a Neuroglancer precomputed dataset.

    Parameters
    ----------
    img_path : str
        Path to be checked (can be local, GCS, or S3).

    Returns
    -------
    bool
        True if the path appears to be a Neuroglancer precomputed dataset.
    """
    try:
        # Build kvstore spec
        bucket_name, path = util.parse_cloud_path(img_path)
        kv = {"driver": "gcs", "bucket": bucket_name, "path": path}

        # Open the info file
        store = ts.KvStore.open(kv).result()
        raw = store.read(b"info").result()

        # Only proceed if the key exists and has content
        if raw.state != "missing" and raw.value:
            info = json.loads(raw.value.decode("utf8"))
            is_valid_type = info.get("type") in ("image", "segmentation")
            if isinstance(info, dict) and is_valid_type and "scales" in info:
                return True
        return False
    except Exception:
        return False


def plot_mips(img, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Image to generate MIPs from.
    vmax : None or float
        Brightness used as upper limit of the colormap. Default is None.
    """
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        mip = np.max(img, axis=i)
        axs[i].imshow(mip, vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()


def to_physical(voxel, anisotropy, offset=(0, 0, 0)):
    """
    Converts a voxel coordinate to a physical coordinate by applying the
    anisotropy scaling factors.

    Parameters
    ----------
    voxel : ArrayLike
        Voxel coordinate to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    offset : Tuple[int], optional
        Shift to be applied to "voxel". Default is (0, 0, 0).

    Returns
    -------
    Tuple[float]
        Physical coordinate.
    """
    voxel = voxel[::-1]
    return tuple([voxel[i] * anisotropy[i] - offset[i] for i in range(3)])


def to_voxels(xyz, anisotropy):
    """
    Converts coordinate from a physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordinate to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.

    Returns
    -------
    Tuple[int]
        Voxel coordinate.
    """
    voxel = [int(xyz[i] / anisotropy[i]) for i in range(3)]
    return tuple(voxel[::-1])
