"""
Created on Thu May 21 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Custom dataset classes.

"""

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
        node_spacing=1
    ):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.anisotropy = anisotropy
        self.min_cable_length = min_cable_length
        self.node_spacing = node_spacing

        # Core data structures
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

    def __getitem__(self):
        pass
