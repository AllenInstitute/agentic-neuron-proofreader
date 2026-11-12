"""The geometric merge walk: one GT neuron at a time and moved to a branching node, as in
``segmentation_skeleton_metrics.MergeCountMetric``, against the earlier pooled walk."""
import unittest

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from agentic_neuron_proofreader.data_modules import canonical_labeling as cl
from agentic_neuron_proofreader.data_modules.canonical_labeling import MergeThresholds

SPACING = 4.0
THRESHOLDS = MergeThresholds.nodes(12)


def chain(start, stop, fixed, axis):
    """Points every ``SPACING`` um from ``start`` to ``stop`` along ``axis`` (0 = x, 1 = y)."""
    values = np.arange(start, stop + 1e-6, SPACING)
    points = np.zeros((len(values), 3))
    points[:, axis], points[:, 1 - axis] = values, fixed
    return points


class Gt(nx.Graph):
    def __init__(self, neurons=()):
        super().__init__()
        if not len(neurons):            # networkx builds an empty instance for subgraph views
            return
        self.node_xyz = np.concatenate(neurons)
        self.node_component_id = np.repeat(np.arange(len(neurons)), [len(n) for n in neurons])
        self.component_id_to_swc_id = {i: f"neuron-{i}" for i in range(len(neurons))}
        self.add_nodes_from(range(len(self.node_xyz)))
        offset = 0
        for neuron in neurons:
            self.add_edges_from((offset + i, offset + i + 1) for i in range(len(neuron) - 1))
            offset += len(neuron)
        self.kdtree = cKDTree(self.node_xyz)

    def node_segment_id(self, node):
        return self.component_id_to_swc_id[int(self.node_component_id[node])]


class Fragment(nx.Graph):
    def __init__(self, xyz=(), edges=(), segment=5):
        super().__init__()
        self.node_xyz = np.asarray(xyz, dtype=float)
        self.add_nodes_from(range(len(xyz)))
        self.add_edges_from(edges)
        self.node_component_id = np.zeros(len(xyz), dtype=np.int64)
        self.component_id_to_swc_id = {0: f"{segment}.0"}


def two_neurons_joined():
    """Neuron 0 runs along x (y = 0), neuron 1 along y (x = 200) from 20 um above its end; one
    fragment follows the first and then the second, so both of its leaves lie on a tracing."""
    gt = Gt([chain(0, 200, 0.0, 0), chain(20, 220, 200.0, 1)])
    xyz = np.concatenate([chain(0, 200, 0.0, 0), chain(4, 220, 200.0, 1)])
    fragment = Fragment(xyz, [(i, i + 1) for i in range(len(xyz) - 1)])
    return fragment, gt, np.full(len(gt.node_xyz), 5, dtype=np.int64)


def traced_shaft_with_untraced_branch():
    """One neuron along x; the fragment follows it and carries a 120 um branch at x = 100."""
    gt = Gt([chain(0, 200, 0.0, 0)])
    shaft, branch = chain(0, 200, 0.0, 0), chain(4, 120, 100.0, 1)
    xyz = np.concatenate([shaft, branch])
    edges = [(i, i + 1) for i in range(len(shaft) - 1)]
    edges += [(25, len(shaft))] + [(len(shaft) + i, len(shaft) + i + 1) for i in range(len(branch) - 1)]
    return Fragment(xyz, edges), gt, np.full(len(gt.node_xyz), 5, dtype=np.int64)


class MergeWalkTests(unittest.TestCase):
    def test_two_traced_neurons_joined_by_one_fragment(self):
        fragment, gt, labels = two_neurons_joined()
        pooled = cl.geometric_merge_sites(fragment, gt, labels, verbose=False, thresholds=THRESHOLDS,
                                          walk=cl.WALK_POOLED)
        self.assertEqual(pooled, (set(), []))          # no leaf is far from every tracing
        merged, sites = cl.geometric_merge_sites(fragment, gt, labels, verbose=False, thresholds=THRESHOLDS)
        self.assertEqual(merged, {5})
        self.assertEqual(len(sites), 1)                # one walk per neuron, 12 um apart, de-duplicated
        self.assertEqual(sites[0]["gt_neuron"], "neuron-0")
        self.assertEqual(sites[0]["xyz"], (200.0, 4.0, 0.0))
        raw = cl._per_neuron_merge_sites(fragment, gt, labels, 12, verbose=False)
        self.assertEqual([(s["gt_neuron"], s["xyz"]) for s in raw],
                         [("neuron-0", (200.0, 4.0, 0.0)), ("neuron-1", (200.0, 16.0, 0.0))])

    def test_site_moves_to_the_branching_node(self):
        fragment, gt, labels = traced_shaft_with_untraced_branch()
        _, pooled = cl.geometric_merge_sites(fragment, gt, labels, verbose=False, thresholds=THRESHOLDS,
                                             walk=cl.WALK_POOLED)
        self.assertEqual([s["xyz"] for s in pooled], [(100.0, 4.0, 0.0)])      # where the walk stopped
        _, sites = cl.geometric_merge_sites(fragment, gt, labels, verbose=False, thresholds=THRESHOLDS)
        self.assertEqual([s["xyz"] for s in sites], [(100.0, 0.0, 0.0)])       # the junction on the shaft
        self.assertEqual(fragment.degree[25], 3)

    def test_pass_through_and_unknown_walk(self):
        fragment, gt, labels = traced_shaft_with_untraced_branch()
        short = labels.copy()
        short[:20], short[31:] = 7, 7                   # the segment labels 11 GT nodes around the branch
        self.assertEqual(cl.geometric_merge_sites(fragment, gt, short, verbose=False, thresholds=THRESHOLDS),
                         (set(), []))
        short[19] = 5                                   # 12 nodes: no longer a pass-through
        self.assertEqual(len(cl.geometric_merge_sites(fragment, gt, short, verbose=False,
                                                      thresholds=THRESHOLDS)[1]), 1)
        with self.assertRaises(ValueError):
            cl.geometric_merge_sites(fragment, gt, labels, verbose=False, walk="neuron")

    def test_refresh_records_the_walk(self):
        from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

        gt = Gt([chain(0, 200, 0.0, 0)])
        dataset = BrainDataset("fragments", "gt", "image", _skip_load=True)
        dataset.gt_graph = gt
        gt.node_label = np.full(len(gt.node_xyz), 5, dtype=np.int64)
        dataset.refresh_merge_labels(verbose=False)
        self.assertEqual(gt.label_policy["walk"], cl.WALK_PER_NEURON)
        dataset.refresh_merge_labels(verbose=False, walk=cl.WALK_POOLED)
        self.assertEqual(gt.label_policy["walk"], cl.WALK_POOLED)


if __name__ == "__main__":
    unittest.main()
