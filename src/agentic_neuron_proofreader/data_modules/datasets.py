"""
Created on Thu May 21 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom dataset classes.

"""

import pickle

from torch.utils.data import Dataset

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
        _skip_load=False,
    ):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.fragments_path = fragments_path
        self.gt_path = gt_path
        self.img_path = img_path
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
            "anisotropy": self.anisotropy,
            "min_cable_length": self.min_cable_length,
            "node_spacing": self.node_spacing,
            "fragments_graph": self.fragments_graph,
            "gt_graph": self.gt_graph,
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
            _skip_load=True,
        )
        instance.fragments_graph = payload["fragments_graph"]
        instance.gt_graph = payload["gt_graph"]
        instance.img = img_util.TensorStoreImage(payload["img_path"])
        return instance

    def __getitem__(self):
        pass
