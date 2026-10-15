import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from agentic_neuron_proofreader.data_modules import canonical_labeling as cl
from agentic_neuron_proofreader.data_modules.canonical_labeling import MergeThresholds


class GtGraph(nx.Graph):
    """Two GT neurons, each a straight chain with a fixed node spacing."""

    def __init__(self, nodes_per_neuron=30, spacing=4.0):
        super().__init__()
        n = nodes_per_neuron
        a = np.column_stack([np.arange(n) * spacing, np.zeros(n), np.zeros(n)])
        b = np.column_stack([np.zeros(n), np.arange(n) * spacing + 500.0, np.zeros(n)])
        self.node_xyz = np.concatenate([a, b])
        self.node_component_id = np.repeat([0, 1], n)
        self.component_id_to_swc_id = {0: "neuron-a", 1: "neuron-b"}
        self.add_nodes_from(range(2 * n))
        self.add_edges_from((i, i + 1) for i in range(n - 1))
        self.add_edges_from((n + i, n + i + 1) for i in range(n - 1))

    def node_segment_id(self, node):
        return self.component_id_to_swc_id[int(self.node_component_id[node])]


class MergeThresholdTests(unittest.TestCase):
    def test_defaults_equal_the_module_constants(self):
        t = MergeThresholds()
        self.assertEqual((t.min_nodes, t.passthru_min_cc, t.policy), (cl.MERGE_MIN_NODES, cl.MERGE_PASSTHRU_MIN_CC, "nodes"))
        self.assertIs(cl.resolve_thresholds(None), cl.DEFAULT_THRESHOLDS)
        self.assertEqual(MergeThresholds.nodes(7), MergeThresholds(min_nodes=7, passthru_min_cc=7))
        with self.assertRaises(ValueError):
            MergeThresholds(min_nodes=2.5)

    def test_from_length_uses_the_graph_node_spacing(self):
        t = MergeThresholds.from_length(GtGraph(spacing=4.0), 50.0)
        self.assertEqual((t.min_nodes, t.passthru_min_cc, t.policy, t.length_um), (12, 12, "length", 50.0))
        self.assertAlmostEqual(t.gt_median_edge_um, 4.0)
        self.assertEqual(MergeThresholds.from_length(GtGraph(spacing=3.9), 50.0).min_nodes, 13)
        self.assertEqual(MergeThresholds.from_length(GtGraph(spacing=1.0), 50.0).min_nodes, 50)
        self.assertEqual(MergeThresholds.from_length(GtGraph(spacing=400.0), 50.0).min_nodes, 1)
        self.assertEqual(set(t.as_record()), {"min_nodes", "passthru_min_cc", "policy", "length_um", "gt_median_edge_um"})

    def test_node_count_rule_edge_error_and_summary_follow_the_thresholds(self):
        gt = GtGraph(nodes_per_neuron=30)
        labels = np.full(60, 5, dtype=np.int64)          # one segment covers 30 nodes of each neuron
        self.assertEqual(cl.merge_labels(gt, labels), set())                                   # 30 <= 50
        self.assertEqual(cl.merge_labels(gt, labels, thresholds=MergeThresholds.nodes(12)), {5})
        self.assertEqual(cl.merge_labels(gt, labels, 12), {5})                                # positional still works
        err_default = cl.compute_edge_error(gt, labels)
        err_scaled = cl.compute_edge_error(gt, labels, thresholds=MergeThresholds.nodes(12))
        self.assertEqual(int((err_default == cl.EDGE_MERGED).sum()), 0)
        self.assertEqual(int((err_scaled == cl.EDGE_MERGED).sum()), gt.number_of_edges())
        self.assertEqual(cl.summarize(gt, labels)["total_merges"], 0)
        self.assertEqual(cl.summarize(gt, labels, thresholds=MergeThresholds.nodes(12))["total_merges"], 1)
        # an explicit merge set with a too-high count threshold still yields no merge events
        self.assertEqual(cl.summarize(gt, labels, merge_label_set={5})["total_merges"], 0)

    def test_passthru_test_uses_the_threshold(self):
        comps = [{1, 2, 3}]
        self.assertTrue(cl._is_nonmerge_passthru(2, comps))                 # 3 < 50
        self.assertFalse(cl._is_nonmerge_passthru(2, comps, min_cc=3))      # 3 < 3 is False
        self.assertTrue(cl._is_nonmerge_passthru(9, comps, min_cc=3))       # not in any component

    def test_refresh_records_the_policy_and_the_cache_round_trips_it(self):
        from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

        gt = GtGraph(nodes_per_neuron=30)
        dataset = BrainDataset("fragments", "gt", "image", _skip_load=True)
        dataset.gt_graph = gt
        gt.node_label = np.full(60, 5, dtype=np.int64)
        dataset.refresh_merge_labels(verbose=False)
        np.testing.assert_array_equal(gt.merge_labels, [])
        self.assertEqual(gt.label_policy["policy"], "nodes")
        self.assertEqual(gt.merge_site_metadata["thresholds"]["min_nodes"], 50)
        scaled = MergeThresholds.from_length(gt, 50.0)
        dataset.refresh_merge_labels(verbose=False, thresholds=scaled)
        np.testing.assert_array_equal(gt.merge_labels, [5])
        self.assertEqual(gt.label_policy["min_nodes"], 12)
        self.assertEqual(gt.label_policy["policy"], "length")
        self.assertFalse(gt.label_policy["reused_geometric_sites"])
        self.assertIn("written", gt.label_policy)
        with tempfile.TemporaryDirectory() as directory, \
                patch("agentic_neuron_proofreader.data_modules.datasets.img_util.TensorStoreImage"):
            path = Path(directory) / "cache_add.pkl"
            dataset.save(path)
            loaded = BrainDataset.load_from_cache(path)
        self.assertEqual(loaded.gt_graph.label_policy, gt.label_policy)
        np.testing.assert_array_equal(loaded.gt_graph.merge_labels, [5])


if __name__ == "__main__":
    unittest.main()
