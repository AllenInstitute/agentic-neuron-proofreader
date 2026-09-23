"""
Created on Thu May 21 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom dataset classes.

"""

import pickle

import numpy as np
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
    def label_gt_from_segmentation(self, segmentation_path=None,
                                   geometric_merges=True, verbose=True):
        """
        Reads the dense predicted segmentation at every GT node's voxel and
        attaches the canonical per-node label and per-edge error class to
        ``self.gt_graph`` (as ``node_label`` and ``edge_error``).

        This is the one cloud-dependent step; run it ONCE before ``save`` so
        every later load is cache-only. It mirrors
        ``segmentation_skeleton_metrics``' label-then-fix-misalignments
        procedure, ported onto this package's ``SkeletonGraph`` so the stored
        arrays are indexed by *this* graph's node ids.

        Merge labels come from the UNION of the node-count rule
        (``merge_labels``) and, when ``geometric_merges`` is True, the geometric
        fragment walk (``geometric_merge_sites``) -- together these reproduce the
        canonical ``labels_with_merge`` set, including grazing fusions the
        node-count rule alone misses. The walk uses ``fragments_graph`` +
        ``gt_graph.kdtree`` (both cache-resident, no extra cloud read).
        The stored merge_sites list combines that walk with two-GT junction
        localization. Site counts therefore need not match the original
        geometric-only MergeCountMetric.

        Parameters
        ----------
        segmentation_path : str, optional
            Path to the dense predicted-segmentation volume. Falls back to
            ``self.segmentation_path``. NOTE: this is the segmentation, not
            ``img_path`` (the raw fused image).
        geometric_merges : bool, optional
            Run the geometric merge walk over ``fragments_graph``. Default True.
            Set False to skip it. Two-GT junction localization still runs.
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

        self.segmentation_path = seg_path
        self.gt_graph.node_label = node_label
        self.refresh_merge_labels(geometric_merges=geometric_merges, verbose=verbose)
        return node_label

    def refresh_merge_labels(self, geometric_merges=True, reuse_geometric_sites=False, verbose=True):
        """Rebuild merge truth from stored GT node labels, with no volume reads.

        Both criteria feed one merge_labels set and one merge_sites list. When
        reusing a labelled cache, preserve all old geometric sites and merge
        labels; rebuild only the supplemental two-GT site localization.
        """
        if self.gt_graph is None or getattr(self.gt_graph, "node_label", None) is None:
            raise ValueError("Canonical GT node labels are required")
        node_label = self.gt_graph.node_label
        merge_set = canonical_labeling.merge_labels(self.gt_graph, node_label)
        sites = []
        if reuse_geometric_sites:
            if not geometric_merges:
                raise ValueError("Cannot reuse geometric sites with geometric_merges=False")
            stored_sites = getattr(self.gt_graph, "merge_sites", None)
            stored_labels = getattr(self.gt_graph, "merge_labels", None)
            if stored_sites is None or stored_labels is None:
                raise ValueError("Stored geometric sites and merge labels are required")
            sites = stored_sites
            merge_set.update(int(label) for label in stored_labels)
        elif geometric_merges and self.fragments_graph is not None:
            self.gt_graph.set_kdtree()
            geometric_labels, sites = canonical_labeling.geometric_merge_sites(
                self.fragments_graph, self.gt_graph, node_label, verbose=verbose)
            merge_set.update(geometric_labels)
        metadata = None
        if self.fragments_graph is not None:
            sites, metadata = canonical_labeling.combined_merge_sites(
                self.fragments_graph, self.gt_graph, node_label, sites, verbose=verbose)
        merge_set.update(int(site["segment_id"]) for site in sites)
        self.gt_graph.edge_error = canonical_labeling.compute_edge_error(
            self.gt_graph, node_label, merge_label_set=merge_set)
        self.gt_graph.merge_labels = np.array(sorted(merge_set), dtype=np.int64)
        self.gt_graph.merge_sites = sites
        self.gt_graph.merge_site_metadata = metadata
        self.gt_graph.junction_gt_audit = None
        return sites

    def audit_junction_gt_connections(self, junction_nodes, **parameters):
        """Attach opt-in, versioned review evidence without relabeling the cache.

        Requires canonical GT node labels already in memory. No cloud reads,
        training, or changes to merge_labels / merge_sites are performed.
        Repeated calls replace the previous audit with the requested node set.
        """
        if self.fragments_graph is None or self.gt_graph is None:
            raise ValueError("Both fragment and GT graphs are required")
        labels = getattr(self.gt_graph, "node_label", None)
        if labels is None:
            raise ValueError("Canonical GT node labels are required")
        audit = canonical_labeling.audit_junction_gt_connections(
            self.fragments_graph, self.gt_graph, labels, junction_nodes, **parameters
        )
        self.gt_graph.junction_gt_audit = audit
        return audit

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
            # Merge segment ids (union of node-count rule + geometric walk) and
            # combined geometric-walk and two-GT junction sites.
            "gt_merge_labels": getattr(self.gt_graph, "merge_labels", None),
            "gt_merge_sites": getattr(self.gt_graph, "merge_sites", None),
            "gt_merge_site_metadata": getattr(self.gt_graph, "merge_site_metadata", None),
            "gt_junction_audit": getattr(self.gt_graph, "junction_gt_audit", None),
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
        merge_labels = payload.get("gt_merge_labels")
        if merge_labels is not None:
            instance.gt_graph.merge_labels = merge_labels
        merge_sites = payload.get("gt_merge_sites")
        if merge_sites is not None:
            instance.gt_graph.merge_sites = merge_sites
        site_metadata = payload.get("gt_merge_site_metadata")
        if site_metadata is not None:
            instance.gt_graph.merge_site_metadata = site_metadata
        junction_audit = payload.get("gt_junction_audit")
        if junction_audit is not None:
            instance.gt_graph.junction_gt_audit = junction_audit
        return instance

    def __getitem__(self):
        pass
