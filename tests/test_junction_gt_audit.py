import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from agentic_neuron_proofreader.data_modules.canonical_labeling import (
    audit_junction_gt_connections, combined_merge_sites,
)


class AuditGraph(nx.Graph):
    def node_segment_id(self, node):
        return self.component_id_to_swc_id[int(self.node_component_id[node])].split(".")[0]


def crossing_graphs(same_neuron=False, second_segment=111):
    fragment = AuditGraph()
    fragment.node_xyz = np.array([[0, 0, 0], [60, 0, 0], [-60, 0, 0],
                                  [0, 60, 0], [0, -60, 0]], dtype=float)
    fragment.node_component_id = np.zeros(5, dtype=int)
    fragment.component_id_to_swc_id = {0: "111.0"}
    fragment.add_edges_from((0, node) for node in range(1, 5))
    gt = AuditGraph()
    positions = np.arange(-60., 61.)
    zeros = np.zeros_like(positions)
    gt.node_xyz = np.concatenate([np.column_stack([positions, zeros, zeros]),
                                  np.column_stack([zeros, positions, zeros])])
    gt.node_component_id = np.repeat([0, 1], len(positions))
    gt.component_id_to_swc_id = {0: "neuron-a", 1: "neuron-a" if same_neuron else "neuron-b"}
    gt.add_nodes_from(range(len(gt.node_xyz)))
    labels = np.repeat([111, second_segment], len(positions))
    return fragment, gt, labels


class JunctionAuditTests(unittest.TestCase):
    def test_full_labeling_generates_both_site_sources(self):
        from agentic_neuron_proofreader.data_modules import canonical_labeling
        from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

        fragment, gt, labels = crossing_graphs()
        dataset = BrainDataset("fragments", "gt", "image", segmentation_path="segmentation", _skip_load=True)
        dataset.fragments_graph, dataset.gt_graph = fragment, gt
        legacy = [{"segment_id": 222, "gt_neuron": "neuron-a", "xyz": (999., 0., 0.)}]
        with patch("agentic_neuron_proofreader.data_modules.datasets.img_util.TensorStoreImage"), \
                patch.object(canonical_labeling, "label_gt_nodes", return_value=labels), \
                patch.object(canonical_labeling, "fix_label_misalignments"), \
                patch.object(gt, "set_kdtree", create=True), \
                patch.object(canonical_labeling, "geometric_merge_sites", return_value=({222}, legacy)) as walk:
            dataset.label_gt_from_segmentation(verbose=False)
        walk.assert_called_once()
        self.assertEqual({site["source"] for site in gt.merge_sites}, {"geometric_walk", "two_gt_junction"})
        np.testing.assert_array_equal(gt.merge_labels, [111, 222])
        np.testing.assert_array_equal(gt.node_label, labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache_add.pkl"
            dataset.save(path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["gt_merge_sites"], gt.merge_sites)
            self.assertEqual(payload["gt_merge_site_metadata"]["n_combined_sites"], 2)

    def test_site_consolidation_uses_cable_distance_and_neuron_identity(self):
        from agentic_neuron_proofreader.data_modules import canonical_labeling

        fragment, gt, labels = crossing_graphs()
        fragment.node_xyz = np.array([[0., 0., 0.], [10., 0., 0.], [60., 0., 0.],
                                      [5., 0., 0.], [0., 5., 0.]])
        fragment.clear_edges()
        fragment.add_edges_from([(0, 1), (1, 2), (0, 3)])
        records = [{"node_id": node, "segment_id": 111, "xyz": fragment.node_xyz[node].tolist(),
                    "gt_neurons": ["a", "b"] if node != 3 else ["a", "c"],
                    "branches": [], "status": "merge_supported"} for node in range(5)]
        with patch.object(canonical_labeling, "audit_junction_gt_connections", return_value={
                "method": "test", "parameters": {}, "junctions": records}):
            sites, _ = canonical_labeling.junction_merge_sites(fragment, gt, labels, verbose=False)
        self.assertEqual([site["node_id"] for site in sites], [0, 2, 3, 4])

    def test_combined_sites_preserve_geometric_and_are_idempotent(self):
        fragment, gt, labels = crossing_graphs()
        old = [{"segment_id": 222, "gt_neuron": "untraced-partner", "xyz": (999., 0., 0.)}]
        sites, metadata = combined_merge_sites(fragment, gt, labels, old, verbose=False)
        self.assertEqual(len(sites), 2)
        self.assertEqual(sites[0]["xyz"], old[0]["xyz"])
        self.assertEqual(sites[0]["source"], "geometric_walk")
        self.assertEqual(sites[1]["source"], "two_gt_junction")
        self.assertEqual(sites[1]["node_id"], 0)
        self.assertEqual(metadata["n_raw_junctions_audited"], 1)
        rerun, repeated = combined_merge_sites(fragment, gt, labels, sites, verbose=False)
        self.assertEqual(rerun, sites)
        self.assertEqual(repeated, metadata)
        self.assertNotIn("source", old[0])

    def test_combined_sites_coalesce_exact_overlap_and_preserve_old_positive(self):
        fragment, gt, labels = crossing_graphs()
        old = [{"segment_id": 111, "gt_neuron": "neuron-a", "xyz": (0., 0., 0.)}]
        sites, _ = combined_merge_sites(fragment, gt, labels, old, verbose=False)
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["sources"], ["geometric_walk", "two_gt_junction"])
        self.assertEqual(combined_merge_sites(fragment, gt, labels, sites, verbose=False)[0], sites)
        fragment, gt, labels = crossing_graphs(same_neuron=True)
        sites, _ = combined_merge_sites(fragment, gt, labels, old, verbose=False)
        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0]["source"], "geometric_walk")

    def test_refresh_preserves_legacy_labels_without_cloud_reads(self):
        from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

        fragment, gt, labels = crossing_graphs()
        gt.node_label = labels
        gt.merge_sites = [{"segment_id": 222, "gt_neuron": "neuron-a", "xyz": (999., 0., 0.)}]
        gt.merge_labels = np.array([111, 222])
        dataset = BrainDataset("fragments", "gt", "image", _skip_load=True)
        dataset.fragments_graph, dataset.gt_graph = fragment, gt
        with patch("agentic_neuron_proofreader.data_modules.datasets.img_util.TensorStoreImage",
                   side_effect=AssertionError("No volume access")):
            dataset.refresh_merge_labels(reuse_geometric_sites=True, verbose=False)
        np.testing.assert_array_equal(gt.node_label, labels)
        np.testing.assert_array_equal(gt.merge_labels, [111, 222])
        self.assertEqual(len(gt.merge_sites), 2)
        self.assertEqual(gt.merge_site_metadata["n_combined_sites"], 2)

    def test_dataset_roundtrip_preserves_audit_and_legacy_labels(self):
        from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

        fragment, gt, labels = crossing_graphs()
        gt.node_label = labels
        gt.merge_labels = np.array([111])
        gt.merge_sites = [{"segment_id": 111, "gt_neuron": "neuron-a", "xyz": [999., 0., 0.]}]
        dataset = BrainDataset("fragments", "gt", "image", _skip_load=True)
        dataset.fragments_graph, dataset.gt_graph = fragment, gt
        audit = dataset.audit_junction_gt_connections([0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audited.pkl"
            dataset.save(path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["gt_junction_audit"], audit)
            self.assertEqual(payload["gt_merge_sites"], gt.merge_sites)
            np.testing.assert_array_equal(payload["gt_merge_labels"], [111])
            with patch("agentic_neuron_proofreader.data_modules.datasets.img_util.TensorStoreImage"):
                restored = BrainDataset.load_from_cache(path)
            self.assertEqual(restored.gt_graph.junction_gt_audit, audit)
            self.assertEqual(restored.gt_graph.merge_sites, gt.merge_sites)

    def test_two_traced_neurons_supported_without_old_site(self):
        fragment, gt, labels = crossing_graphs()
        before = labels.copy()
        result = audit_junction_gt_connections(fragment, gt, labels, [0])
        self.assertTrue(result["audit_only"])
        self.assertEqual(result["schema_version"], 1)
        record = result["junctions"][0]
        self.assertEqual(record["status"], "merge_supported")
        self.assertEqual(record["gt_neurons"], ["neuron-a", "neuron-b"])
        self.assertTrue(all(branch["n_samples"] == 25 for branch in record["branches"]))
        np.testing.assert_array_equal(labels, before)
        self.assertFalse(hasattr(gt, "merge_sites"))

    def test_same_neuron_or_foreign_segment_does_not_prove_merge(self):
        for options in ({"same_neuron": True}, {"second_segment": 222}):
            with self.subTest(options=options):
                fragment, gt, labels = crossing_graphs(**options)
                result = audit_junction_gt_connections(fragment, gt, labels, [0])
                self.assertEqual(result["junctions"][0]["status"], "unknown")

    def test_competing_neurons_are_ambiguous(self):
        fragment, gt, labels = crossing_graphs()
        gt.node_xyz[121:] = gt.node_xyz[:121] + [0, 0, 1]
        result = audit_junction_gt_connections(fragment, gt, labels, [0])
        self.assertEqual(result["junctions"][0]["status"], "unknown")

    def test_short_arms_and_sparse_gt_remain_unknown(self):
        fragment, gt, labels = crossing_graphs()
        fragment.node_xyz *= 0.2
        result = audit_junction_gt_connections(fragment, gt, labels, [0])
        self.assertEqual(result["junctions"][0]["status"], "unknown")
        fragment, gt, labels = crossing_graphs()
        labels[125:] = 0
        result = audit_junction_gt_connections(fragment, gt, labels, [0])
        self.assertEqual(result["junctions"][0]["status"], "unknown")

    def test_reconverging_arms_do_not_prove_local_merge(self):
        fragment, gt, labels = crossing_graphs()
        fragment.add_edge(1, 3)
        result = audit_junction_gt_connections(fragment, gt, labels, [0], branch_length_um=150.)
        self.assertEqual(result["junctions"][0]["status"], "unknown")
        self.assertEqual(result["junctions"][0]["reason"], "reconverging_arms")

    def test_stops_at_next_junction(self):
        fragment, gt, labels = crossing_graphs()
        fragment.node_xyz = np.vstack([fragment.node_xyz,
                                        [[0, 15, 0], [0, -15, 0], [0, 15, 3], [0, -15, 3]]])
        fragment.node_component_id = np.zeros(len(fragment.node_xyz), dtype=int)
        fragment.remove_edges_from([(0, 3), (0, 4)])
        fragment.add_edges_from([(0, 5), (5, 3), (5, 7), (0, 6), (6, 4), (6, 8)])
        result = audit_junction_gt_connections(fragment, gt, labels, [0])
        self.assertEqual(result["junctions"][0]["status"], "unknown")
        self.assertEqual([branch["extent_um"] for branch in result["junctions"][0]["branches"][-2:]],
                         [15., 15.])

    def test_empty_gt_stays_unknown(self):
        fragment, gt, _ = crossing_graphs()
        gt.clear()
        gt.node_xyz = np.empty((0, 3))
        gt.node_component_id = np.empty(0, dtype=int)
        result = audit_junction_gt_connections(fragment, gt, np.empty(0, dtype=int), [0])
        self.assertEqual(result["junctions"][0]["status"], "unknown")

    def test_invalid_nodes_and_parameters_rejected(self):
        fragment, gt, labels = crossing_graphs()
        with self.assertRaises(ValueError):
            audit_junction_gt_connections(fragment, gt, labels, [1])
        with self.assertRaises(ValueError):
            audit_junction_gt_connections(fragment, gt, labels, [0], sample_step_um=0)


if __name__ == "__main__":
    unittest.main()