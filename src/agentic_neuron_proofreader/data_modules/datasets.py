"""
Created on Thu May 21 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom dataset classes.

"""

import pickle

from torch.utils.data import Dataset

from agentic_neuron_proofreader.data_modules import canonical_labeling
from agentic_neuron_proofreader.data_modules.graph_classes import SkeletonGraph
from agentic_neuron_proofreader.utils import img_util


class BrainDataset(Dataset):

    def __init__(
        self,
        fragments_path,
        gt_path,
        img_path,
        anisotropy=(1.0, 1.0, 1.0),
        min_cable_length=0,
        node_spacing=1,
        segmentation_path=None,
        _skip_load=False,
    ):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.fragments_path = fragments_path
        self.gt_path = gt_path
        self.img_path = img_path
        self.segmentation_path = segmentation_path
        self.anisotropy = anisotropy
        self.min_cable_length = min_cable_length
        self.node_spacing = node_spacing

        # Core data structures (skipped when reconstructing from cache)
        if _skip_load:
            self.fragments_graph = None
            self.gt_graph = None
            self.img = None
        else:
            self.fragments_graph = self.create_graph(fragments_path)
            self.gt_graph = self.create_graph(gt_path)
            self.img = img_util.TensorStoreImage(img_path)

    # --- Constructor Helpers ---
    def create_graph(self, skels_path):
        graph = SkeletonGraph(
            anisotropy=self.anisotropy,
            min_cable_length=self.min_cable_length,
            node_spacing=self.node_spacing,
        )
        graph.load(skels_path)
        return graph

    # --- Canonical labeling ---
    def label_gt_from_segmentation(self, segmentation_path=None, verbose=True):
        """
        Reads the dense predicted segmentation at every GT node's voxel and
        attaches the canonical per-node label and per-edge error class to
        ``self.gt_graph`` (as ``node_label`` and ``edge_error``).

        This is the one cloud-dependent step; run it ONCE before ``save`` so
        every later load is cache-only. It mirrors
        ``segmentation_skeleton_metrics``' label-then-fix-misalignments
        procedure, ported onto this package's ``SkeletonGraph`` so the stored
        arrays are indexed by *this* graph's node ids.

        Parameters
        ----------
        segmentation_path : str, optional
            Path to the dense predicted-segmentation volume. Falls back to
            ``self.segmentation_path``. NOTE: this is the segmentation, not
            ``img_path`` (the raw fused image).
        verbose : bool, optional
            Show progress while reading. Default True.

        Returns
        -------
        numpy.ndarray
            The per-node label array (also stored on ``self.gt_graph``).
        """
        seg_path = segmentation_path or self.segmentation_path
        if seg_path is None:
            raise ValueError(
                "segmentation_path is required to compute canonical labels "
                "(this is the dense segmentation volume, not img_path)."
            )
        if self.gt_graph is None:
            raise ValueError("gt_graph is not loaded.")

        segmentation = img_util.TensorStoreImage(seg_path)
        node_label = canonical_labeling.label_gt_nodes(
            self.gt_graph, segmentation, verbose=verbose
        )
        canonical_labeling.fix_label_misalignments(self.gt_graph, node_label)
        edge_error = canonical_labeling.compute_edge_error(
            self.gt_graph, node_label
        )

        self.segmentation_path = seg_path
        self.gt_graph.node_label = node_label
        self.gt_graph.edge_error = edge_error
        return node_label

    # --- Persistence ---
    def save(self, path):
        """
        Pickles the two skeleton graphs (the slow-to-build part) and the paths
        / parameters needed to re-open the lazy image readers. The
        TensorStore-backed `img` is not pickled because it re-instantiates
        instantly from `img_path`.

        Parameters
        ----------
        path : str
            Local path to write the cache file to.
        """
        payload = {
            "fragments_path": self.fragments_path,
            "gt_path": self.gt_path,
            "img_path": self.img_path,
            "segmentation_path": self.segmentation_path,
            "anisotropy": self.anisotropy,
            "min_cable_length": self.min_cable_length,
            "node_spacing": self.node_spacing,
            "fragments_graph": self.fragments_graph,
            "gt_graph": self.gt_graph,
            # Canonical GT labels (None until label_gt_from_segmentation ran).
            # Stored as top-level arrays as well as on gt_graph so a reader can
            # grab them without depending on the graph's attribute layout.
            "gt_node_canonical_label": getattr(
                self.gt_graph, "node_label", None
            ),
            "gt_edge_error": getattr(self.gt_graph, "edge_error", None),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_cache(cls, path):
        """
        Loads a `BrainDataset` previously saved with `save`. The two skeleton
        graphs are restored from the pickle; `img` is rebuilt from `img_path`.

        Parameters
        ----------
        path : str
            Local path to a cache file written by `save`.

        Returns
        -------
        BrainDataset
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(
            payload["fragments_path"],
            payload["gt_path"],
            payload["img_path"],
            anisotropy=payload["anisotropy"],
            min_cable_length=payload["min_cable_length"],
            node_spacing=payload["node_spacing"],
            segmentation_path=payload.get("segmentation_path"),
            _skip_load=True,
        )
        instance.fragments_graph = payload["fragments_graph"]
        instance.gt_graph = payload["gt_graph"]
        instance.img = img_util.TensorStoreImage(payload["img_path"])

        # Restore canonical labels onto gt_graph when present (older caches and
        # un-relabeled caches simply lack these keys / store None).
        node_label = payload.get("gt_node_canonical_label")
        if node_label is not None:
            instance.gt_graph.node_label = node_label
        edge_error = payload.get("gt_edge_error")
        if edge_error is not None:
            instance.gt_graph.edge_error = edge_error
        return instance

    def __getitem__(self):
        pass
