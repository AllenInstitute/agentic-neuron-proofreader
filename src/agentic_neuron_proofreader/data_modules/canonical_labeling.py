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

# Minimum GT nodes a neuron must carry of a shared segment for that segment to
# count as a merge against it. Mirrors the ``num_nodes > 50`` test in
# ``segmentation_skeleton_metrics`` ``MergedEdgePercentMetric.detect_label_intersections``
# -- it filters incidental segment grazes from genuine fusions.
MERGE_MIN_NODES = 50

# Geometric merge-walk thresholds (mirror ``MergeCountMetric`` in
# ``segmentation_skeleton_metrics``). A merge is found by walking a fragment from
# a leaf that sits far from the GT, inward until it re-approaches the GT.
MERGE_DIST_AWAY_UM = 50.0   # leaf must be this far from any GT node to start a walk
MERGE_APPROACH_UM = 6.0     # walk inward until within this distance of a GT node
MERGE_PASSTHRU_MIN_CC = 50  # GT same-label component smaller than this = pass-through
MERGE_DEDUP_UM = 30.0       # merge sites closer than this are duplicates


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


def segment_neuron_node_counts(gt_graph, node_label):
    """
    Maps each predicted segment id -> {GT neuron name: number of GT nodes of
    that segment on that neuron}.

    This node-count detail is what lets us apply the canonical ``> 50 nodes``
    merge threshold (a segment grazing a neuron at a few nodes is not a merge).
    """
    seg_to_neuron_counts = defaultdict(lambda: defaultdict(int))
    for node in gt_graph.nodes:
        lab = int(node_label[node])
        if lab != UNLABELED:
            seg_to_neuron_counts[lab][gt_graph.node_segment_id(node)] += 1
    return seg_to_neuron_counts


def segment_to_gt_neurons(gt_graph, node_label):
    """
    Maps each predicted segment id -> set of GT neuron names it lands on
    (any number of nodes). Used for split counting, where every distinct
    segment touching a neuron matters.
    """
    return {
        lab: set(counts)
        for lab, counts in segment_neuron_node_counts(gt_graph, node_label).items()
    }


def merge_labels(gt_graph, node_label, min_nodes=MERGE_MIN_NODES):
    """
    Set of segment ids that constitute a merge by the *node-count* rule: a
    segment landing on >= 2 distinct GT neurons with **more than ``min_nodes``
    GT nodes on each**.

    Mirrors ``MergedEdgePercentMetric.detect_label_intersections`` (the
    ``num_nodes1 > 50 and num_nodes2 > 50`` test). This catches fusions where the
    segment lays down many GT nodes on both neurons, but MISSES fusions whose
    bridge runs through empty space and only grazes the second neuron -- those are
    found instead by the geometric walk in ``geometric_merge_labels``.
    """
    labels = set()
    for lab, counts in segment_neuron_node_counts(gt_graph, node_label).items():
        substantial = [n for n, c in counts.items() if c > min_nodes]
        if len(substantial) >= 2:
            labels.add(lab)
    return labels


# --- Geometric merge walk (port of segmentation_skeleton_metrics.MergeCountMetric) ---
#
# The node-count rule above only sees WHERE GT nodes landed. The canonical merge
# metric also uses the fragment's PHYSICAL PATH: it walks each fragment from a
# leaf that sits far from the GT, inward until it re-approaches the GT center-line;
# if that approach lands on a different GT neuron than the fragment's own segment
# label (and is not a tiny pass-through), the fragment fuses two neurons -> merge.
#
# Both inputs are in the cache: fragments_graph (geometry + segment ids) and
# gt_graph.kdtree / node_label. No segmentation read is needed.


def _fragment_components(fragments_graph, only_labels=None):
    """
    Groups fragment node ids by connected component (one component = one fragment
    skeleton). If ``only_labels`` is given, yields only components whose segment
    id is in that set -- the canonical walk only inspects fragments whose label
    actually appears on the GT, so this prunes the ~hundreds-of-thousands of
    fragments down to the few that can possibly merge GT neurons.

    Yields ``(segment_id, node_ids_list)``.
    """
    comp_to_nodes = defaultdict(list)
    node_comp = fragments_graph.node_component_id
    for n in fragments_graph.nodes:
        comp_to_nodes[int(node_comp[n])].append(n)
    for comp_id, nodes in comp_to_nodes.items():
        seg_id = int(fragments_graph.component_id_to_swc_id[comp_id].split(".")[0])
        if only_labels is None or seg_id in only_labels:
            yield seg_id, nodes


def _label_to_gt_nodes(gt_graph, node_label):
    """One O(N) pass: label -> list of GT node ids carrying it (skips 0)."""
    out = defaultdict(list)
    arr = np.asarray(node_label)
    for n in gt_graph.nodes:
        lab = int(arr[n])
        if lab != UNLABELED:
            out[lab].append(n)
    return out


def _gt_label_components(gt_graph, label, label_nodes):
    """
    Connected components (as node-id sets) of the GT subgraph whose nodes carry
    ``label``. ``label_nodes`` is the precomputed label -> node-id list, so this
    avoids re-scanning all GT nodes per fragment label.
    """
    nodes = label_nodes.get(label, [])
    if not nodes:
        return []
    return list(nx.connected_components(gt_graph.subgraph(nodes)))


def _is_nonmerge_passthru(gt_node, comps):
    """True if ``gt_node`` sits in a GT same-label component smaller than
    ``MERGE_PASSTHRU_MIN_CC`` -- a likely pass-through, not a merge. ``comps`` is
    the same-label component list from ``_gt_label_components``."""
    for cc in comps:
        if gt_node in cc:
            return len(cc) < MERGE_PASSTHRU_MIN_CC
    return True


def geometric_merge_sites(fragments_graph, gt_graph, node_label, verbose=True):
    """
    Walks fragment skeletons to locate merge sites, porting
    ``MergeCountMetric.search_for_merges`` / ``find_merge_site`` / ``verify_site``
    onto the cache graphs.

    For each fragment whose segment label appears on the GT, start at any leaf
    that is > ``MERGE_DIST_AWAY_UM`` from the GT, walk inward until within
    ``MERGE_APPROACH_UM`` of a GT node; if that node is not a tiny same-label
    pass-through, record a merge site. Spatially-redundant sites within
    ``MERGE_DEDUP_UM`` are collapsed.

    Returns
    -------
    (merge_label_set, sites)
        ``merge_label_set`` -- segment ids flagged as merges by the walk.
        ``sites`` -- list of dicts (segment_id, gt_neuron, xyz) for inspection.
    """
    gt_kdtree = gt_graph.kdtree
    label_nodes = _label_to_gt_nodes(gt_graph, node_label)
    labels_on_gt = set(label_nodes)
    frag_xyz = fragments_graph.node_xyz

    comps = list(_fragment_components(fragments_graph, only_labels=labels_on_gt))
    iterator = comps
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(comps, desc="Merge walk")
        except ImportError:
            pass

    raw_sites = list()
    for seg_id, nodes in iterator:
        # Cache the GT same-label components once per fragment label.
        gt_comps = _gt_label_components(gt_graph, seg_id, label_nodes)
        node_set = set(nodes)
        # Degree within this fragment component (for leaf detection).
        leaves = [n for n in nodes if fragments_graph.degree[n] == 1]
        visited = set()
        for leaf in leaves:
            if leaf in visited:
                continue
            visited.add(leaf)
            dist, _ = gt_kdtree.query(frag_xyz[leaf])
            if dist <= MERGE_DIST_AWAY_UM:
                continue
            # Walk inward (DFS) until we re-approach the GT.
            site = _walk_to_merge_site(
                fragments_graph, gt_graph, node_label, gt_kdtree,
                leaf, node_set, visited, seg_id, gt_comps, frag_xyz,
            )
            if site is not None:
                raw_sites.append(site)

    sites = _dedup_sites(raw_sites)
    merge_label_set = {s["segment_id"] for s in sites}
    return merge_label_set, sites


def _walk_to_merge_site(fragments_graph, gt_graph, node_label, gt_kdtree,
                        source, node_set, visited, seg_id, gt_comps, frag_xyz):
    """DFS inward from ``source`` until a node lands within ``MERGE_APPROACH_UM``
    of the GT; verify and return a site dict, or None."""
    queue = deque([source])
    visited.add(source)
    while queue:
        i = queue.pop()
        dist_i, gt_node = gt_kdtree.query(frag_xyz[i])
        if dist_i < MERGE_APPROACH_UM:
            if _is_nonmerge_passthru(int(gt_node), gt_comps):
                return None
            xyz = frag_xyz[i]
            return {
                "segment_id": seg_id,
                "gt_neuron": gt_graph.node_segment_id(int(gt_node)),
                "xyz": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            }
        for j in fragments_graph.neighbors(i):
            if j in node_set and j not in visited:
                queue.append(j)
                visited.add(j)
    return None


def _dedup_sites(raw_sites):
    """Collapse merge sites closer than ``MERGE_DEDUP_UM`` (mirrors
    ``remove_repeat_merge_sites``)."""
    if not raw_sites:
        return []
    from scipy.spatial import KDTree

    pts = np.array([s["xyz"] for s in raw_sites])
    kdtree = KDTree(pts)
    rm = set()
    for i, s in enumerate(raw_sites):
        if i in rm:
            continue
        for j in kdtree.query_ball_point(s["xyz"], MERGE_DEDUP_UM):
            if j != i:
                rm.add(j)
    return [s for k, s in enumerate(raw_sites) if k not in rm]


def compute_edge_error(gt_graph, node_label, merge_min_nodes=MERGE_MIN_NODES,
                       merge_label_set=None):
    """
    Derives a per-edge error class from the node labels alone (no extra reads).

    Returns a ``(E,)`` uint8 array parallel to ``list(gt_graph.edges)``:

    - ``EDGE_OMIT``    (2): either endpoint unlabeled (the canonical OR rule).
    - ``EDGE_SPLIT``   (1): both labeled, different segment ids.
    - ``EDGE_MERGED``  (3): both endpoints carry the same segment id, and that
      segment is a merge label.
    - ``EDGE_CORRECT`` (0): same segment id, not a merge label.

    ``merge_label_set``, if given, is the set of merge segment ids to use --
    normally the UNION of the node-count rule (``merge_labels``) and the geometric
    walk (``geometric_merge_sites``), which together reproduce canonical
    ``labels_with_merge``. When omitted, only the node-count rule is applied (it
    needs no fragments graph), which under-flags fusions whose bridge only grazes
    the second neuron.

    Storing this is pure convenience -- it is reproducible from ``node_label``
    (plus the fragments graph for the geometric part) -- so a viewer can color the
    skeleton with zero logic.
    """
    merged = (merge_label_set if merge_label_set is not None
              else merge_labels(gt_graph, node_label, merge_min_nodes))
    err = np.empty(gt_graph.number_of_edges(), dtype=np.uint8)
    for k, (i, j) in enumerate(gt_graph.edges):
        li, lj = int(node_label[i]), int(node_label[j])
        if li == UNLABELED or lj == UNLABELED:
            err[k] = EDGE_OMIT
        elif li != lj:
            err[k] = EDGE_SPLIT
        elif li in merged:
            err[k] = EDGE_MERGED
        else:
            err[k] = EDGE_CORRECT
    return err


def summarize(gt_graph, node_label, edge_error=None, merge_label_set=None):
    """
    Computes the cache-only canonical summary: per-edge class fractions, total
    splits, and total merges. Useful for the verification step.

    ``merge_label_set`` -- the merge segment ids (normally the union of the
    node-count rule and the geometric walk). When omitted, only the node-count
    rule is used, which under-counts grazing fusions (see ``compute_edge_error``).

    Returns a dict of plain Python numbers.
    """
    if edge_error is None:
        edge_error = compute_edge_error(gt_graph, node_label,
                                        merge_label_set=merge_label_set)

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

    # Merges. Canonical "% Merged Edges" counts (nodes_with_label_on_neuron - 1)
    # per merge label per neuron over labels_with_merge (= union of the
    # node-count rule and the geometric walk), NOT raw graph edges. The
    # EDGE_MERGED array is a per-edge view for visualization; this percentage is
    # what matches results.csv.
    seg_neuron_counts = segment_neuron_node_counts(gt_graph, node_label)
    merged = (merge_label_set if merge_label_set is not None
              else merge_labels(gt_graph, node_label))
    num_merged_edges = 0
    for lab in merged:
        for neuron, c in seg_neuron_counts.get(lab, {}).items():
            num_merged_edges += max(c - 1, 0)
    # # Merges: a merge label fuses k neurons (with > MERGE_MIN_NODES each) ->
    # (k - 1) merge events.
    total_merges = sum(
        max(len([n for n, c in seg_neuron_counts.get(lab, {}).items()
                 if c > MERGE_MIN_NODES]) - 1, 0)
        for lab in merged
    )

    return {
        "num_edges": E,
        "pct_split_edges": 100.0 * counts["split"] / E if E else 0.0,
        "pct_omit_edges": 100.0 * counts["omit"] / E if E else 0.0,
        "pct_merged_edges": 100.0 * num_merged_edges / E if E else 0.0,
        "pct_merged_edges_per_edge_view": 100.0 * counts["merged"] / E if E else 0.0,
        "pct_correct_edges": 100.0 * counts["correct"] / E if E else 0.0,
        "total_splits": total_splits,
        "total_merges": total_merges,
        "num_neurons": len(neuron_to_segs),
    }
