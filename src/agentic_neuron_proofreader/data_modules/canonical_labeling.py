"""
Canonical ground-truth labeling utilities.

These helpers bake the *canonical* per-GT-node error labels into a cache so that,
after loading, the split / merge / omit classification is available directly --
no nearest-fragment proxy, no free match-tolerance parameter, and no second
read of the dense segmentation volume.

The canonical label of a GT node is the segment id of the predicted
segmentation *at that node's voxel*. This mirrors
``segmentation_skeleton_metrics``' ``GraphLoader._label_graph`` followed by
``LabeledGraph.fix_label_misalignments`` -- ported here so it can run against
the cache's own ``SkeletonGraph`` (whose node layout is what the stored arrays
must be indexed by).

Reading the segmentation is the one cloud-dependent step and is intended to run
ONCE at build / relabel time. Everything stored afterwards is read straight from
the pickle.

Axis convention
---------------
The cache ``SkeletonGraph`` stores ``node_xyz`` in (x, y, z) microns and exposes
``node_voxel(i)`` -> (z, y, x) integer voxel (``xyz / anisotropy`` then reversed,
see ``img_util.to_voxels``). We index the segmentation patch in that same
(z, y, x) order. The exact orientation the cloud segmentation expects can only be
confirmed by matching the resulting metrics against the canonical
``results.csv`` -- the verification step in ``scripts/relabel_cache.py`` is the
gate for that, and ``swap_zyx_to_xyz`` / ``transpose`` below exist to adjust it
without touching anything else.
"""

from collections import defaultdict, deque

import networkx as nx
import numpy as np

# Background / "no segment here" sentinel. Canonical code uses the string "0";
# we store an integer array, so 0 plays the same role.
UNLABELED = 0

# Edge-error codes stored in `gt_edge_error` (parallel to list(gt_graph.edges)).
EDGE_CORRECT = 0
EDGE_SPLIT = 1
EDGE_OMIT = 2
EDGE_MERGED = 3
EDGE_ERROR_NAMES = {
    EDGE_CORRECT: "correct",
    EDGE_SPLIT: "split",
    EDGE_OMIT: "omit",
    EDGE_MERGED: "merged",
}


def label_gt_nodes(
    gt_graph,
    segmentation,
    batch_size=128,
    batch_dist=128.0,
    verbose=True,
):
    """
    Reads the predicted segment id at every GT node's voxel and returns one
    integer label per node (0 = unlabeled / background).

    Ported from ``segmentation_skeleton_metrics`` ``GraphLoader._label_graph``:
    nodes are gathered into spatially-local batches by walking the graph
    (DFS), one segmentation patch is read per batch, and each node is indexed
    out of that patch. Reading per-batch instead of per-node keeps the number
    of cloud reads small.

    Parameters
    ----------
    gt_graph : SkeletonGraph
        The cache's ground-truth graph. Its ``node_voxel(i)`` convention
        defines how the returned array is indexed.
    segmentation : img_util.TensorStoreImage
        Reader over the dense predicted-segmentation volume.
    batch_size : int, optional
        Maximum nodes per segmentation read. Default 128.
    batch_dist : float, optional
        Start a new batch once a node is farther than this (microns) from the
        batch root, so each patch stays compact. Default 128.
    verbose : bool, optional
        Show a progress bar. Default True.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` int64 array; ``node_label[i]`` is the segment id at node i's
        voxel, or ``UNLABELED`` (0) if the segmentation reads background there.
    """
    num_nodes = gt_graph.number_of_nodes()
    node_label = np.zeros(num_nodes, dtype=np.int64)

    # Group nodes into compact batches by walking the graph.
    batches = _spatial_batches(gt_graph, batch_size, batch_dist)
    iterator = batches
    if verbose:
        try:
            from tqdm import tqdm

            iterator = tqdm(batches, desc="Label GT nodes")
        except ImportError:
            pass

    for batch in iterator:
        _label_batch(gt_graph, batch, segmentation, node_label)

    return node_label


def _spatial_batches(gt_graph, batch_size, batch_dist):
    """
    Walks each connected component (DFS) accumulating spatially-local node
    batches, mirroring the batching in the canonical ``_label_graph``.
    """
    batches = list()
    batch = list()
    root = None
    for i, j in nx.dfs_edges(gt_graph):
        if not batch:
            root = i
            batch.append(i)
        too_far = gt_graph.dist(root, j) > batch_dist
        if too_far or len(batch) >= batch_size:
            batches.append(batch)
            batch = [j]
            root = j
        else:
            batch.append(j)
    if batch:
        batches.append(batch)

    # Singleton components (no edges) are never visited by dfs_edges; sweep up
    # any node that did not land in a batch.
    seen = {n for b in batches for n in b}
    for n in gt_graph.nodes:
        if n not in seen:
            batches.append([n])
    return batches


def _label_batch(gt_graph, batch, segmentation, node_label):
    """
    Reads one segmentation patch covering ``batch`` and writes each node's
    segment id into ``node_label``.

    ``img_util.TensorStoreImage.read(voxel, shape)`` treats ``voxel`` as the
    patch *center* (``start = voxel - shape // 2``). To make the patch start
    exactly at ``bbox_min`` we therefore pass ``center = bbox_min + shape // 2``
    -- the ``shape // 2`` cancels and ``start == bbox_min`` exactly, so node
    voxels index in as ``voxel - bbox_min``.
    """
    voxels = np.array([gt_graph.node_voxel(i) for i in batch], dtype=np.int64)
    bbox_min = voxels.min(axis=0)
    bbox_max = voxels.max(axis=0) + 1
    shape = (bbox_max - bbox_min).astype(np.int64)
    center = bbox_min + shape // 2

    patch = np.asarray(
        segmentation.read(
            tuple(int(c) for c in center), tuple(int(s) for s in shape)
        )
    )
    # The reader may return leading singleton axes (e.g. (1, 1, dz, dy, dx)).
    # Reshape to the 3 spatial dims rather than squeeze -- squeeze would also
    # collapse a genuinely size-1 spatial axis (when a batch is planar along
    # one axis) and misalign the indexing below.
    patch = patch.reshape(tuple(int(s) for s in shape))

    local = voxels - bbox_min
    for node, (a, b, c) in zip(batch, local):
        node_label[node] = int(patch[a, b, c])


def fix_label_misalignments(gt_graph, node_label):
    """
    Heals thin unlabeled gaps that are segmentation/skeleton misalignments
    rather than true omits, mirroring ``LabeledGraph.fix_label_misalignments``.

    A maximal run of ``0`` nodes that is bordered on all sides by exactly one
    nonzero segment id is relabeled to that id (it was an alignment slip, not a
    real gap). Runs bordered by two or more distinct ids are left as ``0``.

    Mutates and returns ``node_label``.
    """
    visited_edges = set()
    for i, j in deque(nx.dfs_edges(gt_graph)):
        if frozenset((i, j)) in visited_edges:
            continue
        if int(node_label[j]) == UNLABELED:
            _check_misalignment(gt_graph, node_label, visited_edges, i, j)
        visited_edges.add(frozenset((i, j)))
    return node_label


def _check_misalignment(gt_graph, node_label, visited_edges, nb, root):
    """Flood the zero-region at ``root``; relabel it iff exactly one nonzero
    segment id borders it."""
    label_collisions = set()
    queue = deque([root])
    visited = set()
    while queue:
        j = queue.popleft()
        label_j = int(node_label[j])
        if label_j != UNLABELED:
            label_collisions.add(label_j)
        visited.add(j)
        if label_j == UNLABELED:
            for k in gt_graph.neighbors(j):
                if k not in visited:
                    if frozenset((j, k)) not in visited_edges or k == nb:
                        queue.append(k)
                        visited_edges.add(frozenset((j, k)))

    if len(label_collisions) == 1:
        label = label_collisions.pop()
        for node in visited:
            if int(node_label[node]) == UNLABELED:
                node_label[node] = label


def segment_to_gt_neurons(gt_graph, node_label):
    """
    Maps each predicted segment id -> set of GT neuron names it lands on.

    A segment touching >= 2 distinct GT neurons is a merge; this mapping is the
    dual used to flag merged edges and count merges.
    """
    seg_to_neurons = defaultdict(set)
    for node in gt_graph.nodes:
        lab = int(node_label[node])
        if lab != UNLABELED:
            seg_to_neurons[lab].add(gt_graph.node_segment_id(node))
    return seg_to_neurons


def compute_edge_error(gt_graph, node_label):
    """
    Derives a per-edge error class from the node labels alone (no extra reads).

    Returns a ``(E,)`` uint8 array parallel to ``list(gt_graph.edges)``:

    - ``EDGE_OMIT``    (2): either endpoint unlabeled (the canonical OR rule).
    - ``EDGE_SPLIT``   (1): both labeled, different segment ids.
    - ``EDGE_MERGED``  (3): both endpoints carry the same segment id, but that
      segment also touches another GT neuron (i.e. it fuses neurons).
    - ``EDGE_CORRECT`` (0): same segment id, segment touches only this neuron.

    Storing this is pure convenience -- it is fully reproducible from
    ``node_label`` -- so a viewer can color the skeleton with zero logic.
    """
    seg_to_neurons = segment_to_gt_neurons(gt_graph, node_label)
    err = np.empty(gt_graph.number_of_edges(), dtype=np.uint8)
    for k, (i, j) in enumerate(gt_graph.edges):
        li, lj = int(node_label[i]), int(node_label[j])
        if li == UNLABELED or lj == UNLABELED:
            err[k] = EDGE_OMIT
        elif li != lj:
            err[k] = EDGE_SPLIT
        elif len(seg_to_neurons[li]) >= 2:
            err[k] = EDGE_MERGED
        else:
            err[k] = EDGE_CORRECT
    return err


def summarize(gt_graph, node_label, edge_error=None):
    """
    Computes the cache-only canonical summary: per-edge class fractions, total
    splits, and total merges. Useful for the verification step.

    Returns a dict of plain Python numbers.
    """
    if edge_error is None:
        edge_error = compute_edge_error(gt_graph, node_label)

    E = len(edge_error)
    counts = {name: int((edge_error == code).sum())
              for code, name in EDGE_ERROR_NAMES.items()}

    # Splits per neuron = (#distinct segments touching it) - 1.
    neuron_to_segs = defaultdict(set)
    for node in gt_graph.nodes:
        lab = int(node_label[node])
        if lab != UNLABELED:
            neuron_to_segs[gt_graph.node_segment_id(node)].add(lab)
    total_splits = sum(max(len(s) - 1, 0) for s in neuron_to_segs.values())

    # Merges: each segment touching k>=2 GT neurons contributes (k-1).
    seg_to_neurons = segment_to_gt_neurons(gt_graph, node_label)
    total_merges = sum(len(ns) - 1 for ns in seg_to_neurons.values() if len(ns) >= 2)

    return {
        "num_edges": E,
        "pct_split_edges": 100.0 * counts["split"] / E if E else 0.0,
        "pct_omit_edges": 100.0 * counts["omit"] / E if E else 0.0,
        "pct_merged_edges": 100.0 * counts["merged"] / E if E else 0.0,
        "pct_correct_edges": 100.0 * counts["correct"] / E if E else 0.0,
        "total_splits": total_splits,
        "total_merges": total_merges,
        "num_neurons": len(neuron_to_segs),
    }
