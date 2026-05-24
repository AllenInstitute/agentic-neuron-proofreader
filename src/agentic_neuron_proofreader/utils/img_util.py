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

from matplotlib.colors import ListedColormap

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


def make_segmentation_colormap(mask, seed=42):
    """
    Creates a matplotlib ListedColormap for a segmentation. Ensures label 0
    maps to black and all other labels get distinct random colors.

    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask with integer labels. Assumes label 0 is background.
    seed : int, optional
        Random seed for color reproducibility. Default is 42.

    Returns
    -------
    ListedColormap
        Colormap with black for background and unique colors for other labels.
    """
    n_labels = int(mask.max()) + 1
    rng = np.random.default_rng(seed)
    colors = [(0, 0, 0)]
    colors += list(rng.uniform(0.2, 1.0, size=(n_labels - 1, 3)))
    return ListedColormap(colors)


def plot_segmentation_mips(segmentation):
    """
    Plots maximum intensity projections (MIPs) of a segmentation.

    Parameters
    ----------
    segmentation : numpy.ndarray
        Segmentation to generate MIPs from.
    """
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    cmap = make_segmentation_colormap(segmentation)
    for i in range(3):
        mip = np.max(segmentation, axis=i)
        axs[i].imshow(mip, cmap=cmap, interpolation="none")
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_skeleton_mips(node_groups, patch_shape, separate_rows=False):
    """
    Plots skeleton-node MIPs along the XY, XZ, and YZ axes for the same patch
    used by `plot_mips` / `plot_segmentation_mips`. Each group can include
    edges, which are drawn as line segments connecting their endpoints.

    Parameters
    ----------
    node_groups : Dict[str, dict]
        Mapping from group label (e.g., "GT", "UNet") to a dict with:
            - "nodes" : numpy.ndarray of shape (N, 3) -- local (z, y, x)
              voxel coords. Use `SkeletonGraph.nodes_in_patch(...)`.
            - "color" : str, matplotlib color used for nodes (and edges,
              when `components` is not supplied).
            - "edges" : numpy.ndarray of shape (E, 2, 3), optional. Each
              entry is [start_xyz, end_xyz] in local (z, y, x) voxels.
              Use `SkeletonGraph.edges_in_patch(...)`.
            - "components" : numpy.ndarray of shape (E,), optional. Per-edge
              connected-component IDs. When supplied, edges are colored by
              component using a categorical colormap (tab20) instead of the
              group's `color`. Use `edges_in_patch(..., return_components=True)`.
        Legacy form `(coords, color)` tuples are also accepted.
    patch_shape : Tuple[int]
        Shape of the patch in (z, y, x) order. Used to fix axis limits so the
        skeleton MIP panels align with image / segmentation MIPs of the same
        volume.
    separate_rows : bool, optional
        If True, render each group on its own row of the figure (no overlay).
        Otherwise all groups are drawn on the same 1x3 panel. Default False.
    """
    # Normalize legacy tuple form -> dict
    groups = {}
    for label, value in node_groups.items():
        if isinstance(value, dict):
            groups[label] = value
        else:
            coords, color = value
            groups[label] = {"nodes": coords, "color": color}

    axs_names = ["XY", "XZ", "YZ"]
    # For axis=i MIP in (z, y, x) array, the 2D plane uses the OTHER two axes.
    plane_axes = [(1, 2), (0, 2), (0, 1)]

    n_rows = len(groups) if separate_rows else 1
    fig, axs = plt.subplots(n_rows, 3, figsize=(10, 4 * n_rows), squeeze=False)

    if separate_rows:
        rows = [(r, label, group) for r, (label, group) in enumerate(groups.items())]
    else:
        rows = [(0, label, group) for label, group in groups.items()]

    comp_cmap = plt.get_cmap("tab20")

    for r, label, group in rows:
        coords = group.get("nodes", np.empty((0, 3)))
        color = group["color"]
        edges = group.get("edges", np.empty((0, 2, 3)))
        components = group.get("components")
        # Map each unique component ID -> a slot in tab20 (per-group).
        edge_colors = None
        if components is not None and len(components):
            unique_comps = np.unique(components)
            comp_to_idx = {c: i for i, c in enumerate(unique_comps)}
            edge_colors = np.array(
                [comp_cmap(comp_to_idx[c] % comp_cmap.N) for c in components]
            )
        for i, (name, (a, b)) in enumerate(zip(axs_names, plane_axes)):
            ax = axs[r, i]
            ax.set_facecolor("black")
            if len(edges):
                if edge_colors is not None:
                    # Per-edge color: draw as a LineCollection so we can pass
                    # an array of colors instead of one per ax.plot() call.
                    from matplotlib.collections import LineCollection
                    segs = np.stack([edges[:, 0, [b, a]], edges[:, 1, [b, a]]], axis=1)
                    lc = LineCollection(segs, colors=edge_colors, linewidths=2.0, alpha=0.95)
                    ax.add_collection(lc)
                else:
                    xs = edges[:, :, b].T  # shape (2, E)
                    ys = edges[:, :, a].T
                    ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.95)
            if len(coords):
                ax.scatter(coords[:, b], coords[:, a], s=8, c=color, label=label)
            ax.set_xlim(0, patch_shape[b])
            ax.set_ylim(patch_shape[a], 0)
            ax.set_aspect("equal")
            title = f"{label} -- {name}" if separate_rows else name
            ax.set_title(title, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

    if not separate_rows and any(len(g.get("nodes", [])) for g in groups.values()):
        axs[0, 0].legend(loc="upper right", markerscale=3, fontsize=9)
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
