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

    def __init__(self, graph_config, gt_path, fragments_path, img_path):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.gt_graph = self.create_graph(graph_config, gt_path)
        self.fragments_graph = self.create_graph(graph_config, fragments_path)
        self.img = img_util.TensorStoreImage(img_path)

    # --- Constructor Helpers ---
    def create_graph(self, config, skels_path):
        graph = SkeletonGraph(*config)
        graph.load(skels_path)
        return graph

    def __getitem__(self):
        pass
