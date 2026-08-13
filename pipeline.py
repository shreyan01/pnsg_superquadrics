"""Integration: raw unlabeled cloud -> segmentation -> graph -> multi-part
registry match -> introspective gate -> grasp/ask."""
import numpy as np
from iterative_segment import iterative_two_part_segment
from graph import PartNode, ObjectGraph
from loop import IntrospectiveVocabLoop
from compute_grasp import compute_grasp


def build_graph_from_segmentation(raw_cloud, params_a, params_b, assignment):
    graph = ObjectGraph()
    graph.add_node(PartNode(node_id='n0', params=params_a, role='dominant'))
    if params_b is not None:
        graph.add_node(PartNode(node_id='n1', params=params_b, role='secondary_0'))
    graph.build_edges_mst()
    return graph


def process_scene(raw_cloud, spoken_noun, mu_det, loop: IntrospectiveVocabLoop,
                   feedback_fn, gripper_min_width=0.015, gripper_max_width=0.09,
                   verbose=True, max_iters=8, max_nfev=3000):
    params_a, params_b, assignment = iterative_two_part_segment(
        raw_cloud, verbose=False, max_iters=max_iters, max_nfev=max_nfev)
    graph = build_graph_from_segmentation(raw_cloud, params_a, params_b, assignment)

    if verbose:
        print(graph.describe())

    record = loop.step_graph(graph, spoken_noun, mu_det, feedback_fn)

    if verbose:
        tag = 'ASK ' if record['triggered'] else 'ACT '
        print(f'[{tag}] noun={spoken_noun}  mu_det={mu_det:.2f}  mu_obj={record["mu_obj"]:.2f}  '
              f'chi={record["chi"]:.2f}  alpha={record["alpha_before"]:.2f}  -> {record["registry_action"]}')

    result = {'graph': graph, 'gate_record': record, 'grasp': None}

    if not record['triggered']:
        mode = None
        if record.get('matched_mode'):
            for gm in loop.registry.graph_modes.get(spoken_noun, []):
                if gm.mode_id == record['matched_mode']:
                    mode = gm.part_modes.get('dominant')
                    break
        ranked = compute_grasp(params_a, gripper_min_width, gripper_max_width, mode=mode)
        result['grasp'] = ranked
        if verbose:
            if ranked:
                print(f'      -> grasp: {ranked[0]["approach"]} (score={ranked[0]["score"]:.3f})')
            else:
                print(f'      -> NO FEASIBLE GRASP')

    return result
